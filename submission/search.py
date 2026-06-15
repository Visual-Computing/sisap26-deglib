#!/usr/bin/env python3
"""search.py — SISAP 2026 entrypoint: deglib (DEG + EVP) in the baseline harness.

TIRA invokes this with the standard interface:

    search.py --input $inputDataset/*.h5 \
              --task-description $inputDataset/config.json \
              --output $outputDir

It reads the task config, drives the deglib C++ binary (one graph build, a
``--max-dist`` sweep -> one result file per operating point, reusing the graph),
and writes format-compliant result HDF5 files via ``store_results``.

Scope: Task 1 (k-NN self-join) and Task 2 (MIPS search). Task 3 (sparse SPLADE)
is intentionally skipped cleanly (exit 0) so the spot-check CI stays green.

The compiled binary is located via $DEGLIB_BIN (default /usr/local/bin/deglib_sisap).
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

DEGLIB_BIN = os.environ.get("DEGLIB_BIN", "/usr/local/bin/deglib_sisap")
ALGO = "deglib-evp"
THREADS = os.environ.get("DEGLIB_THREADS", "8")

# --- per-dataset parameter profiles ------------------------------------------
# The graph is built ONCE per profile; max_dist is swept to yield several
# operating points (build/recall trade-off) from that single build. Unknown
# datasets fail fast (see _profile_or_die) rather than silently using bad params.
TASK1_PROFILES = {
    # 1024-dim BGE-M3 (normalised, inner product)
    "wikipedia-small": dict(mode="mode4", non_zeros=576, k_graph=22, k_ext=29,
                            eps_ext=0.001, prune_worst=10, evpK=113,
                            max_dist=[200, 300, 400, 500, 700, 900]),
    "wikipedia":       dict(mode="mode4", non_zeros=608, k_graph=26, k_ext=32,
                            eps_ext=0.002, prune_worst=9, evpK=50,
                            max_dist=[500, 700, 900, 1200, 1400]),
    # 384-dim gooaq spot-check (different family; smoke test only, non_zeros<dim)
    "gooaq-small":     dict(mode="mode4", non_zeros=300, k_graph=24, k_ext=24,
                            eps_ext=0.001, prune_worst=8, evpK=50,
                            max_dist=[200, 400, 800]),
}

# Task 2 (MIPS): graph built once (single-threaded per the rules), then a
# (eps_search x max_dist) sweep yields the operating points. mode5 = L2-build +
# FP16 inner-product search; FLAS pre-sort improves the build.
TASK2_PROFILES = {
    # 128-dim Llama-3 attention (unnormalised inner product)
    "llama-dev":   dict(mode="mode5", k_graph=32, k_ext=64, eps_ext=0.001, build_threads=1,
                        use_flas=True, num_runs=3,
                        max_dist=[5000, 6000, 7000, 8000], eps_search=[0.18, 0.19, 0.2]),
    # spot-check (14k vectors); smoke test only
    "llama-small": dict(mode="mode5", k_graph=32, k_ext=64, eps_ext=0.001, build_threads=1,
                        use_flas=True, num_runs=1,
                        max_dist=[2000, 4000, 8000], eps_search=[0.2, 0.3]),
}


def load_task_config(task_description_path):
    """Load task configuration from a config.json file."""
    with open(task_description_path) as f:
        return json.load(f)


def get_h5_item(f, path):
    """Traverse a slash-separated or list path through an HDF5 file."""
    if isinstance(path, list):
        cur = f
        for p in path:
            cur = cur[p]
        return cur
    cur = f
    for p in path.split("/"):
        cur = cur[p]
    return cur


def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):
    """Write one operating point in the official SISAP result format."""
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['dataset'] = dataset
    f.attrs['task'] = task
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()


def _resolve_input(input_arg):
    """Expand a possibly-globbed --input into the single existing file path.
    Fails fast if zero or more than one file matches (TIRA mounts exactly one)."""
    if any(ch in input_arg for ch in "*?[]"):
        matches = sorted(_glob.glob(input_arg))
        if not matches:
            sys.exit(f"Error: no input file matched {input_arg!r}")
        if len(matches) > 1:
            sys.exit(f"Error: --input {input_arg!r} matched {len(matches)} files, "
                     f"expected exactly one: {matches}")
        return matches[0]
    return input_arg


def _key_path(key):
    """Normalise a config key (str slash-path or list) to an HDF5 path string."""
    return key if isinstance(key, str) else "/".join(key)


def maybe_decompress(input_path, keys_needed):
    """If any needed dataset is chunked/compressed, materialise an uncompressed
    contiguous temp .h5 with just those datasets (the deglib reader cannot read
    gzip/chunked HDF5). Datasets are copied in row blocks to avoid a full-RAM
    peak under the 24 GB limit. Returns (path_for_binary, tmp_path_or_None)."""
    with h5py.File(input_path, "r") as f:
        needs = any(
            (get_h5_item(f, key).chunks is not None
             or get_h5_item(f, key).compression is not None)
            for key in keys_needed
        )
        if not needs:
            return input_path, None
        fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="deglib_in_")
        os.close(fd)
        print(f"[decompress] input is chunked/compressed -> writing uncompressed temp {tmp_path}")
        try:
            with h5py.File(tmp_path, "w") as out:
                for key in keys_needed:
                    src = get_h5_item(f, key)
                    dst = out.create_dataset(_key_path(key), shape=src.shape,
                                             dtype=src.dtype, compression=None, chunks=None)
                    if src.ndim == 0:
                        dst[()] = src[()]
                    else:
                        block = 200000
                        for i in range(0, src.shape[0], block):
                            dst[i:i + block] = src[i:i + block]
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    return tmp_path, tmp_path


def read_op_file(path):
    """Read one deglib operating-point file: header (n,k,T_build,T_explore) then
    uint32 ids (0-based, UINT32_MAX = padding) and float32 distances."""
    with open(path, "rb") as fh:
        n, k = np.frombuffer(fh.read(8), dtype=np.int32)
        n, k = int(n), int(k)
        bt, et = np.frombuffer(fh.read(8), dtype=np.float32)
        ids = np.frombuffer(fh.read(n * k * 4), dtype=np.uint32).reshape(n, k)
        dists = np.frombuffer(fh.read(n * k * 4), dtype=np.float32).reshape(n, k)
    return n, k, float(bt), float(et), ids, dists


def _profile_or_die(profiles, dataset, task):
    profile = profiles.get(dataset)
    if profile is None:
        sys.exit(
            f"Error: no deglib parameter profile for {task} dataset {dataset!r}. "
            f"Known: {sorted(profiles)}. Add a profile before submitting."
        )
    return profile


def _require_binary():
    if not (os.path.isfile(DEGLIB_BIN) and os.access(DEGLIB_BIN, os.X_OK)):
        sys.exit(f"Error: deglib binary not found/executable at {DEGLIB_BIN!r} "
                 f"(set $DEGLIB_BIN).")


def run_task1(input_path, cfg, output_dir):
    dataset = cfg["dataset_name"]
    k = int(cfg.get("k", 15))
    profile = _profile_or_die(TASK1_PROFILES, dataset, "task1")
    _require_binary()
    print(f"[task1] dataset={dataset} mode={profile['mode']} "
          f"non_zeros={profile['non_zeros']} max_dist={profile['max_dist']}")

    bin_input, tmp = maybe_decompress(input_path, ["train"])
    op_dir = None
    try:
        op_dir = tempfile.mkdtemp(prefix="deglib_op_")
        cmd = [
            DEGLIB_BIN, "task1", bin_input, profile["mode"],
            "--threads", THREADS, "--k-top", str(k),
            "--non-zeros", str(profile["non_zeros"]),
            "--k-graph", str(profile["k_graph"]),
            "--k-ext", str(profile["k_ext"]),
            "--eps-ext", str(profile["eps_ext"]),
            "--evpK", str(profile["evpK"]),
            "--max-dist", ",".join(str(m) for m in profile["max_dist"]),
            "--prune-worst", str(profile["prune_worst"]),
            "--no-recall", "--output", op_dir,
        ]
        print("[task1] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        ops = sorted(Path(op_dir).glob("op_*.bin"))
        if not ops:
            sys.exit("Error: deglib produced no operating-point files for task1.")
        sentinel = np.iinfo(np.uint32).max
        for op in ops:
            n, kk, t_build, t_explore, ids, dists = read_op_file(op)
            ids = ids.astype(np.int64)
            # Padding slots (fewer than k candidates) -> map to the node's own
            # 0-based id; after +1 that is a harmless duplicate of the self column,
            # keeping every id a valid 1-based label (and never overflowing int32).
            row0 = np.arange(n, dtype=np.int64)[:, None]
            ids = np.where(ids == sentinel, row0, ids)
            # 0-based ids -> 1-based, then prepend the self-reference at column 0
            # (k+1 columns) to match the ground-truth layout the evaluator uses.
            self_ids = np.arange(1, n + 1, dtype=np.int64)[:, None]
            knns = np.concatenate([self_ids, ids + 1], axis=1).astype(np.int32)
            self_d = np.zeros((n, 1), dtype=np.float32)
            out_d = np.concatenate([self_d, dists], axis=1)
            mobj = re.search(r"op_evpK(\d+)_md(\d+)", op.name)
            evpK, md = mobj.group(1), mobj.group(2)
            params = (f"mode={profile['mode']},non_zeros={profile['non_zeros']},"
                      f"k_graph={profile['k_graph']},k_ext={profile['k_ext']},"
                      f"prune_worst={profile['prune_worst']},evpK={evpK},max_dist={md}")
            # Task 1 is scored on construction time: buildtime = build + explore.
            buildtime = t_build + t_explore
            fn = os.path.join(output_dir, f"deglib_evpK{evpK}_md{md}.h5")
            store_results(fn, ALGO, dataset, "task1", out_d, knns, buildtime, 0.0, params)
            print(f"  wrote {fn}  buildtime={buildtime:.3f}s knns={knns.shape}")
    finally:
        if op_dir is not None:
            shutil.rmtree(op_dir, ignore_errors=True)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def run_task2(input_path, cfg, output_dir):
    dataset = cfg["dataset_name"]
    k = int(cfg.get("k", 30))
    queries_key = cfg.get("queries", "test/queries")
    profile = _profile_or_die(TASK2_PROFILES, dataset, "task2")
    _require_binary()
    print(f"[task2] dataset={dataset} mode={profile['mode']} flas={profile.get('use_flas')} "
          f"eps_search={profile['eps_search']} max_dist={profile['max_dist']}")

    bin_input, tmp = maybe_decompress(input_path, ["train", queries_key])
    op_dir = None
    try:
        op_dir = tempfile.mkdtemp(prefix="deglib_op_")
        cmd = [
            DEGLIB_BIN, "task2", bin_input, profile["mode"],
            "--threads", THREADS, "--build-threads", str(profile["build_threads"]),
            "--k-top", str(k), "--k-graph", str(profile["k_graph"]),
            "--k-ext", str(profile["k_ext"]), "--eps-ext", str(profile["eps_ext"]),
            "--max-dist", ",".join(str(m) for m in profile["max_dist"]),
            "--eps-search", ",".join(str(e) for e in profile["eps_search"]),
            "--num-runs", str(profile["num_runs"]),
            "--no-recall", "--output", op_dir,
        ]
        if profile.get("use_flas"):
            cmd.append("--flas")
        print("[task2] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        ops = sorted(Path(op_dir).glob("op_*.bin"))
        if not ops:
            sys.exit("Error: deglib produced no operating-point files for task2.")
        sentinel = np.iinfo(np.uint32).max
        for op in ops:
            n, kk, t_build, t_search, ids, dists = read_op_file(op)
            # task2 ids are ALREADY 1-based (the binary adds +1 to match test/knns).
            # No self column (queries are separate from the database). Padding slots
            # -> 0, the baseline's "missing" marker (never matches a 1-based id).
            ids = ids.astype(np.int64)
            ids = np.where(ids == sentinel, 0, ids)
            knns = ids.astype(np.int32)
            mobj = re.search(r"op_eps(\d+)_md(\d+)", op.name)
            eps_i, md = mobj.group(1), mobj.group(2)
            eps = int(eps_i) / 1000.0
            params = (f"mode={profile['mode']},k_graph={profile['k_graph']},k_ext={profile['k_ext']},"
                      f"flas={int(bool(profile.get('use_flas')))},num_runs={profile['num_runs']},"
                      f"eps_search={eps},max_dist={md}")
            # Task 2 is scored on search time: querytime = search, buildtime = one-time build.
            fn = os.path.join(output_dir, f"deglib_eps{eps_i}_md{md}.h5")
            store_results(fn, ALGO, dataset, "task2", dists, knns, t_build, t_search, params)
            print(f"  wrote {fn} buildtime={t_build:.3f}s querytime={t_search:.4f}s knns={knns.shape}")
    finally:
        if op_dir is not None:
            shutil.rmtree(op_dir, ignore_errors=True)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to the input HDF5 benchmark file (glob allowed)")
    parser.add_argument("--task-description", required=True,
                        help="Path to the task config JSON file (config.json)")
    parser.add_argument("--output", required=True,
                        help="Directory where result HDF5 files are written")
    args = parser.parse_args()

    cfg = load_task_config(args.task_description)
    task = cfg.get("task")
    os.makedirs(args.output, exist_ok=True)

    # deglib/EVP is a dense method; task3 is sparse SPLADE. Skip cleanly (before
    # touching --input) so the mandatory spot-check CI over all tasks stays green.
    if task == "task3":
        print("task3 (sparse) is not supported by the deglib submission — skipping.",
              file=sys.stderr)
        return 0

    input_path = _resolve_input(args.input)
    if task == "task1":
        run_task1(input_path, cfg, args.output)
    elif task == "task2":
        run_task2(input_path, cfg, args.output)
    else:
        sys.exit(f"Error: unknown task {task!r} in config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
