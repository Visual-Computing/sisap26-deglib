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
ALGO = "deglib"
THREADS = os.environ.get("DEGLIB_THREADS", "8")

# --- per-dataset parameter profiles ------------------------------------------
# Each dataset maps to a LIST of configs; every config is one binary invocation
# (one graph build) that emits one or more operating points via its max_dist (and,
# for task 2, eps_search) sweep. Configs that share build params are grouped into a
# single sweep; configs with distinct build params build a separate graph. Unknown
# datasets fail fast (see _profile_or_die) rather than silently using bad params.
#
# Task 1 config keys: mode, non_zeros, k_graph, k_ext, eps_ext, prune_worst, evpK,
# max_dist (list). The submission slots are the tuned (mode4 + mode7) configs from
# the old python/submission_task1_*.py — 15 operating points per dataset.
TASK1_PROFILES = {
    # 1024-dim BGE-M3 (normalized, inner product) — 200K dev set; 15 tuned slots.
    "wikipedia-small": [
        dict(mode="mode4", non_zeros=768, k_graph=16, k_ext=19, eps_ext=0.001, prune_worst=2,  evpK=28,  max_dist=[576]),  # slot 1  ~0.757
        dict(mode="mode4", non_zeros=704, k_graph=14, k_ext=29, eps_ext=0.001, prune_worst=3,  evpK=31,  max_dist=[707]),  # slot 2  ~0.787
        dict(mode="mode4", non_zeros=704, k_graph=18, k_ext=26, eps_ext=0.001, prune_worst=4,  evpK=37,  max_dist=[404]),  # slot 3  ~0.806
        dict(mode="mode4", non_zeros=768, k_graph=14, k_ext=38, eps_ext=0.001, prune_worst=3,  evpK=37,  max_dist=[666]),  # slot 4  ~0.815
        dict(mode="mode4", non_zeros=640, k_graph=16, k_ext=37, eps_ext=0.001, prune_worst=8,  evpK=38,  max_dist=[620]),  # slot 5  ~0.826
        dict(mode="mode4", non_zeros=704, k_graph=16, k_ext=31, eps_ext=0.001, prune_worst=3,  evpK=86,  max_dist=[431]),  # slot 6  ~0.831
        dict(mode="mode4", non_zeros=384, k_graph=12, k_ext=16, eps_ext=0.001, prune_worst=6,  evpK=95,  max_dist=[742]),  # slot 7  ~0.838
        dict(mode="mode4", non_zeros=768, k_graph=18, k_ext=41, eps_ext=0.001, prune_worst=9,  evpK=78,  max_dist=[586]),  # slot 8  ~0.881
        dict(mode="mode4", non_zeros=576, k_graph=16, k_ext=74, eps_ext=0.001, prune_worst=6,  evpK=48,  max_dist=[561]),  # slot 9  ~0.888
        dict(mode="mode4", non_zeros=576, k_graph=22, k_ext=29, eps_ext=0.001, prune_worst=10, evpK=113, max_dist=[533]),  # slot 10 ~0.894
        dict(mode="mode7", non_zeros=768, k_graph=20, k_ext=18, eps_ext=0.001, prune_worst=10, evpK=41,  max_dist=[220]),  # slot 11 ~0.776
        dict(mode="mode7", non_zeros=704, k_graph=22, k_ext=20, eps_ext=0.001, prune_worst=11, evpK=35,  max_dist=[192]),  # slot 12 ~0.783
        dict(mode="mode7", non_zeros=640, k_graph=18, k_ext=48, eps_ext=0.001, prune_worst=9,  evpK=33,  max_dist=[221]),  # slot 13 ~0.815
        dict(mode="mode7", non_zeros=640, k_graph=26, k_ext=27, eps_ext=0.001, prune_worst=12, evpK=33,  max_dist=[226]),  # slot 14 ~0.840
        dict(mode="mode7", non_zeros=768, k_graph=20, k_ext=39, eps_ext=0.001, prune_worst=10, evpK=77,  max_dist=[384]),  # slot 15 ~0.859
    ],
    # 6.35M BGE-M3 — submission slots (eps_ext=0.002). Slots 1-8 share one build
    # (max_dist sweep); slots 9/10 and 11/15 are tuned individually.
    "wikipedia": [
        dict(mode="mode4", non_zeros=608, k_graph=26, k_ext=32, eps_ext=0.002, prune_worst=9,  evpK=50,  max_dist=[500, 600, 700, 800, 900, 1000, 1200, 1400]),  # slots 1-8
        dict(mode="mode4", non_zeros=512, k_graph=32, k_ext=24, eps_ext=0.002, prune_worst=11, evpK=50,  max_dist=[900]),            # slot 9
        dict(mode="mode4", non_zeros=512, k_graph=32, k_ext=24, eps_ext=0.002, prune_worst=11, evpK=100, max_dist=[800]),            # slot 10
        dict(mode="mode7", non_zeros=576, k_graph=28, k_ext=34, eps_ext=0.002, prune_worst=10, evpK=50,  max_dist=[400]),            # slot 11
        dict(mode="mode7", non_zeros=512, k_graph=32, k_ext=24, eps_ext=0.002, prune_worst=11, evpK=50,  max_dist=[400, 500, 600]),  # slots 12-14
        dict(mode="mode7", non_zeros=576, k_graph=28, k_ext=34, eps_ext=0.002, prune_worst=10, evpK=75,  max_dist=[800]),            # slot 15
    ],
    # 384-dim gooaq spot-check (different family; smoke test only, non_zeros<dim)
    "gooaq-small": [
        dict(mode="mode4", non_zeros=300, k_graph=24, k_ext=24, eps_ext=0.001, prune_worst=8, evpK=50, max_dist=[200, 400, 800]),
    ],
}

# Task 2 (MIPS): each config builds the graph once (single-threaded per the rules),
# then sweeps (eps_search x max_dist). The submission candidates are mode5 (L2-build +
# FP16 inner-product search) and mode7 (L2 d+2 build + FP16 L2 search), both with FLAS.
# Task 2 config keys: mode, k_graph, k_ext, eps_ext, build_threads, use_flas, num_runs,
# max_dist (list), eps_search (list).
TASK2_PROFILES = {
    # 128-dim Llama-3 attention (unnormalized inner product) — submission candidates.
    "llama-dev": [
        dict(mode="mode5", k_graph=32, k_ext=64, eps_ext=0.001, build_threads=1, use_flas=True,
             num_runs=10, max_dist=[6000, 6500, 7000, 7500, 8000, 9000], eps_search=[0.18]),
        dict(mode="mode7", k_graph=32, k_ext=64, eps_ext=0.001, build_threads=1, use_flas=True,
             num_runs=10, max_dist=[5500, 6000, 6250, 6500, 6750, 7000, 8000], eps_search=[0.007]),
    ],
    # spot-check (14k vectors); smoke test only
    "llama-small": [
        dict(mode="mode5", k_graph=32, k_ext=64, eps_ext=0.001, build_threads=1, use_flas=True,
             num_runs=1, max_dist=[2000, 4000, 8000], eps_search=[0.2, 0.3]),
    ],
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
    configs = _profile_or_die(TASK1_PROFILES, dataset, "task1")
    _require_binary()
    print(f"[task1] dataset={dataset}: {len(configs)} config(s) / build(s)")

    bin_input, tmp = maybe_decompress(input_path, ["train"])
    op_root = None
    try:
        op_root = tempfile.mkdtemp(prefix="deglib_op_")
        sentinel = np.iinfo(np.uint32).max
        for ci, c in enumerate(configs):
            # One binary invocation per config = one graph build + its own sweep.
            op_dir = os.path.join(op_root, f"c{ci}")
            os.makedirs(op_dir, exist_ok=True)
            cmd = [
                DEGLIB_BIN, "task1", bin_input, c["mode"],
                "--threads", THREADS, "--k-top", str(k),
                "--non-zeros", str(c["non_zeros"]),
                "--k-graph", str(c["k_graph"]),
                "--k-ext", str(c["k_ext"]),
                "--eps-ext", str(c["eps_ext"]),
                "--evpK", str(c["evpK"]),
                "--max-dist", ",".join(str(m) for m in c["max_dist"]),
                "--prune-worst", str(c["prune_worst"]),
                "--no-recall", "--output", op_dir,
            ]
            print(f"[task1] config {ci + 1}/{len(configs)} ({c['mode']}):", " ".join(cmd))
            subprocess.run(cmd, check=True)

            ops = sorted(Path(op_dir).glob("op_*.bin"))
            if not ops:
                sys.exit(f"Error: deglib produced no operating-point files for task1 config {ci}.")
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
                params = (f"mode={c['mode']},non_zeros={c['non_zeros']},"
                          f"k_graph={c['k_graph']},k_ext={c['k_ext']},"
                          f"prune_worst={c['prune_worst']},evpK={evpK},max_dist={md}")
                # Task 1 is scored on construction time: buildtime = build + explore.
                buildtime = t_build + t_explore
                # Config index keeps filenames unique across builds that happen to
                # share an (evpK, max_dist) pair (e.g. mode4 vs mode7 at the same md).
                fn = os.path.join(output_dir, f"deglib_c{ci}_evpK{evpK}_md{md}.h5")
                store_results(fn, ALGO, dataset, "task1", out_d, knns, buildtime, 0.0, params)
                print(f"  wrote {fn}  buildtime={buildtime:.3f}s knns={knns.shape}")
    finally:
        if op_root is not None:
            shutil.rmtree(op_root, ignore_errors=True)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def run_task2(input_path, cfg, output_dir):
    dataset = cfg["dataset_name"]
    k = int(cfg.get("k", 30))
    queries_key = cfg.get("queries", "test/queries")
    configs = _profile_or_die(TASK2_PROFILES, dataset, "task2")
    _require_binary()
    print(f"[task2] dataset={dataset}: {len(configs)} config(s) / build(s)")

    bin_input, tmp = maybe_decompress(input_path, ["train", queries_key])
    op_root = None
    try:
        op_root = tempfile.mkdtemp(prefix="deglib_op_")
        sentinel = np.iinfo(np.uint32).max
        for ci, c in enumerate(configs):
            # One binary invocation per config = one graph build + its own sweep.
            op_dir = os.path.join(op_root, f"c{ci}")
            os.makedirs(op_dir, exist_ok=True)
            cmd = [
                DEGLIB_BIN, "task2", bin_input, c["mode"],
                "--threads", THREADS, "--build-threads", str(c["build_threads"]),
                "--k-top", str(k), "--k-graph", str(c["k_graph"]),
                "--k-ext", str(c["k_ext"]), "--eps-ext", str(c["eps_ext"]),
                "--max-dist", ",".join(str(m) for m in c["max_dist"]),
                "--eps-search", ",".join(str(e) for e in c["eps_search"]),
                "--num-runs", str(c["num_runs"]),
                "--no-recall", "--output", op_dir,
            ]
            if c.get("use_flas"):
                cmd.append("--flas")
            print(f"[task2] config {ci + 1}/{len(configs)} ({c['mode']}):", " ".join(cmd))
            subprocess.run(cmd, check=True)

            ops = sorted(Path(op_dir).glob("op_*.bin"))
            if not ops:
                sys.exit(f"Error: deglib produced no operating-point files for task2 config {ci}.")
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
                params = (f"mode={c['mode']},k_graph={c['k_graph']},k_ext={c['k_ext']},"
                          f"flas={int(bool(c.get('use_flas')))},num_runs={c['num_runs']},"
                          f"eps_search={eps},max_dist={md}")
                # Task 2 is scored on search time: querytime = search, buildtime = one-time build.
                # Config index keeps filenames unique across builds (e.g. mode5 vs mode7).
                fn = os.path.join(output_dir, f"deglib_c{ci}_eps{eps_i}_md{md}.h5")
                store_results(fn, ALGO, dataset, "task2", dists, knns, t_build, t_search, params)
                print(f"  wrote {fn} buildtime={t_build:.3f}s querytime={t_search:.4f}s knns={knns.shape}")
    finally:
        if op_root is not None:
            shutil.rmtree(op_root, ignore_errors=True)
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
