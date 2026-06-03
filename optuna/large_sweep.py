#!/usr/bin/env python3
"""
large_sweep.py — Build-once + (max_dist x evpK) sweep on the LARGE dataset.

For each curated build-config (carried over from the small Pareto fronts), run
the binary ONCE: it builds the EVP graph a single time and sweeps every
(max_dist, evpK) combination over that in-memory graph. We parse each combo's
recall/time from the log and add the one-time load+quant+build+convert cost to
get the true end-to-end time per (build-config, search-combo).

Rationale: build-graph params translate ~1:1 small->large, but max_dist/evpK do
not — so we reuse the good build settings and re-sweep only the search budget.
`eps_ext` is fixed small (the small report showed it is a non-driver of recall
and a build-time bomb at scale).

Usage:
    python large_sweep.py [--configs id1,id2] [--max-dist 400,800,1400]
                          [--evpK 100,300] [--timeout 5400]
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HOME", "/home/nico/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, "/opt/sisap26-deglib")
from docker_runner import Task1Runner

OUT = Path("/opt/sisap26-deglib/optuna")
RES = OUT / "container_results"
RES.mkdir(parents=True, exist_ok=True)

EPS_EXT = 0.002

# Curated build-configs from the small fronts. Build is identical for mode4 and
# mode7 (both build the EvpBits graph the same way); only explore differs.
# (id, mode, non_zeros, k_graph, k_ext, prune_worst)
# Goal = fastest config reaching recall>=0.8, so we span the CHEAP region
# (low->mid k_graph) and bracket the 0.8 boundary; the slow high-recall tip is
# irrelevant to winning. build is identical for mode4/mode7 (only explore differs).
BUILD_CONFIGS = [
    ("m4_kg12", "mode4", 704, 12, 20, 2),
    ("m4_kg18", "mode4", 704, 18, 26, 4),
    ("m4_kg24", "mode4", 640, 24, 30, 8),
    ("m4_kg32", "mode4", 512, 32, 24, 11),
    ("m4_kg40", "mode4", 512, 40, 31, 14),
    ("m7_kg18", "mode7", 704, 18, 26, 4),
    ("m7_kg32", "mode7", 512, 32, 24, 11),
]

RE = {
    "load":    re.compile(r"Load Time:\s*([\d.]+)"),
    "quant":   re.compile(r"Quantize Time:\s*([\d.]+)"),
    "build":   re.compile(r"Graph Build Time:\s*([\d.]+)"),
    "convert": re.compile(r"Graph Conversion Time:\s*([\d.]+)"),
    "prune":   re.compile(r"Pruning Time:\s*([\d.]+)"),
}
RE_COMBO = re.compile(
    r"evpK=(\d+),\s*max_dist=(\d+)\s*has recall\s*([\d.]+)\s*%\s*and time\s*([\d.]+)\s*s")


def _grab(rx, log):
    m = rx.search(log)
    return float(m.group(1)) if m else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")
    ap.add_argument("--max-dist", default="400,800,1400")
    ap.add_argument("--evpK", default="100,300")
    ap.add_argument("--timeout", type=float, default=5400.0)
    ap.add_argument("--out", default=str(OUT / "large_sweep.csv"))
    args = ap.parse_args()

    wanted = set(args.configs.split(",")) if args.configs else None
    configs = [c for c in BUILD_CONFIGS if (wanted is None or c[0] in wanted)]

    runner = Task1Runner(image_tag="sisap26-deglib", results_dir=RES, echo_logs=True)
    out_path = Path(args.out)
    new_file = not out_path.exists()

    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["config", "mode", "non_zeros", "k_graph", "k_ext", "prune_worst",
                        "eps_ext", "load_s", "quant_s", "build_s", "convert_s",
                        "max_dist", "evpK", "explore_rerank_s", "total_s", "recall"])
        for cid, mode, nz, kg, kext, pw in configs:
            print(f"\n===== {cid} ({mode}) nz={nz} kg={kg} kext={kext} pw={pw} "
                  f"=====", flush=True)
            res = runner.run(mode=mode, size="large", threads=8,
                             non_zeros=nz, k_graph=kg, k_ext=kext, eps_ext=EPS_EXT,
                             prune_worst=pw, max_dist=args.max_dist, evp_k=args.evpK,
                             timeout_s=args.timeout)
            log = "\n".join(res.raw_logs)
            load, quant = _grab(RE["load"], log), _grab(RE["quant"], log)
            build, convert = _grab(RE["build"], log), _grab(RE["convert"], log)
            prune = _grab(RE["prune"], log)
            base = load + quant + build + convert + prune
            combos = list(RE_COMBO.finditer(log))
            if not combos:
                print(f"[{cid}] WARNING: no combo lines parsed (exit={res.exit_code}); "
                      f"base={base:.1f}s", flush=True)
            for m in combos:
                evpk, md = int(m.group(1)), int(m.group(2))
                rec, ct = float(m.group(3)) / 100.0, float(m.group(4))
                total = base + ct
                w.writerow([cid, mode, nz, kg, kext, pw, EPS_EXT,
                            round(load, 1), round(quant, 1), round(build, 1),
                            round(convert, 1), md, evpk, round(ct, 1),
                            round(total, 1), round(rec, 4)])
                f.flush()
                star = "  <== >=0.8" if rec >= 0.8 else ""
                print(f"[{cid}] max_dist={md} evpK={evpk}  recall={rec:.4f}  "
                      f"total={total:.1f}s (base={base:.1f}+{ct:.1f}){star}", flush=True)

    print(f"\nDone -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
