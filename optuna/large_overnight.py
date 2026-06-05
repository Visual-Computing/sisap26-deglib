#!/usr/bin/env python3
"""
large_overnight.py — ~12h build-once sweep to map the LARGE Pareto front densely
and select the best 15 submission candidates (10 mode4 + 5 mode7).

Goal: fastest config reaching Recall@15 >= 0.8 on the 6.35M self-join. Recall is
SIMD-independent, so this AVX-512 run fixes each config's recall (which drives
candidate selection); the final 15 get re-timed on AVX2 later.

Strategy:
  - One container run per (build-config): builds the EVP graph once and sweeps
    the given max_dist x evpK lists over it (search is cheap; build is the cost).
  - Jobs are PRIORITISED so partial completion still yields the essentials:
      Tier A: dense mode4 front (k_graph 22..40 x fine max_dist)   <- most important
      Tier B: mode7 front
      Tier C: evpK trade-off on the boundary configs (cheaper rerank?)
      Tier D: non_zeros sensitivity on the best k_graph (does nz translate?)
  - Incremental CSV writes + a wall-clock budget guard (default 11.5h) so it
    never overruns. Per-run timeout guards pathological builds.

Usage:
    python large_overnight.py [--budget-hours 11.5] [--timeout 4200]
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HOME", "/home/nico/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/sisap26-deglib")
from docker_runner import Task1Runner

OUT = Path("/opt/sisap26-deglib/optuna")
RES = OUT / "container_results"
RES.mkdir(parents=True, exist_ok=True)
EPS_EXT = 0.002

RE = {k: re.compile(p) for k, p in {
    "load": r"Load Time:\s*([\d.]+)", "quant": r"Quantize Time:\s*([\d.]+)",
    "build": r"Graph Build Time:\s*([\d.]+)", "convert": r"Graph Conversion Time:\s*([\d.]+)",
    "prune": r"Pruning Time:\s*([\d.]+)"}.items()}
RE_COMBO = re.compile(
    r"evpK=(\d+),\s*max_dist=(\d+)\s*has recall\s*([\d.]+)\s*%\s*and time\s*([\d.]+)\s*s")


# Default build params per k_graph (interpolated from the small Pareto fronts).
def bp(kg: int) -> tuple[int, int, int]:
    table = {
        22: (672, 28, 7), 24: (640, 30, 8), 26: (608, 32, 9), 28: (576, 34, 10),
        30: (544, 30, 10), 32: (512, 24, 11), 36: (512, 28, 13), 40: (512, 31, 14),
    }
    return table[kg]


FULL_MD = "400,500,600,700,800,900,1000,1200,1400"   # fine boundary grid


def build_jobs() -> list[dict]:
    jobs: list[dict] = []

    def add(tier, mode, kg, md, evpk, nz=None, kext=None, pw=None, tag=""):
        d_nz, d_kext, d_pw = bp(kg)
        jobs.append(dict(
            tier=tier, mode=mode, kg=kg,
            nz=nz or d_nz, kext=kext or d_kext, pw=pw if pw is not None else d_pw,
            md=md, evpk=evpk,
            cid=f"{tier}_{mode[-1]}_kg{kg}{('_' + tag) if tag else ''}"))

    # Tier A — dense mode4 front (knee first so the essentials finish first)
    for kg in (24, 26, 28, 32, 30, 22, 36, 40):
        add("A", "mode4", kg, FULL_MD, "50")
    # Tier B — mode7 front
    for kg in (28, 32, 26, 40):
        add("B", "mode7", kg, FULL_MD, "50")
    # Tier C — evpK trade-off on the boundary configs (cheaper / safer rerank)
    for mode, kg in (("mode4", 24), ("mode4", 28), ("mode4", 32),
                     ("mode7", 28), ("mode7", 32)):
        add("C", mode, kg, "600,800,1000", "20,30,75,100", tag="evpk")
    # Tier D — non_zeros sensitivity on the best k_graph (does nz translate?)
    for kg in (28, 32):
        for nz in (384, 640, 768):
            add("D", "mode4", kg, "600,800,1000", "50", nz=nz, tag=f"nz{nz}")
    return jobs


def grab(rx, log):
    m = rx.search(log)
    return float(m.group(1)) if m else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=11.5)
    ap.add_argument("--timeout", type=float, default=4200.0)
    ap.add_argument("--out", default=str(OUT / "large_overnight.csv"))
    args = ap.parse_args()

    jobs = build_jobs()
    runner = Task1Runner(image_tag="sisap26-deglib", results_dir=RES, echo_logs=False)
    out_path = Path(args.out)
    new_file = not out_path.exists()
    t0 = time.time()
    budget_s = args.budget_hours * 3600

    print(f"[overnight] {len(jobs)} jobs queued; budget={args.budget_hours}h", flush=True)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tier", "config", "mode", "non_zeros", "k_graph", "k_ext",
                        "prune_worst", "eps_ext", "load_s", "quant_s", "build_s",
                        "convert_s", "max_dist", "evpK", "explore_rerank_s",
                        "total_s", "recall"])
        for i, j in enumerate(jobs):
            elapsed = time.time() - t0
            if elapsed > budget_s:
                print(f"[overnight] budget reached after {elapsed/3600:.2f}h; "
                      f"stopping before job {i+1}/{len(jobs)} ({j['cid']}).", flush=True)
                break
            print(f"\n===== [{i+1}/{len(jobs)}] {j['cid']} ({j['mode']}) nz={j['nz']} "
                  f"kg={j['kg']} md=[{j['md']}] evpK=[{j['evpk']}] | "
                  f"elapsed={elapsed/3600:.2f}h =====", flush=True)
            res = runner.run(mode=j["mode"], size="large", threads=8,
                             non_zeros=j["nz"], k_graph=j["kg"], k_ext=j["kext"],
                             eps_ext=EPS_EXT, prune_worst=j["pw"],
                             max_dist=j["md"], evp_k=j["evpk"], timeout_s=args.timeout)
            log = "\n".join(res.raw_logs)
            base = sum(grab(RE[k], log) for k in ("load", "quant", "build", "convert", "prune"))
            combos = list(RE_COMBO.finditer(log))
            if not combos:
                print(f"[{j['cid']}] WARNING: no combos parsed (exit={res.exit_code})", flush=True)
            for m in combos:
                evpk, md = int(m.group(1)), int(m.group(2))
                rec, ct = float(m.group(3)) / 100.0, float(m.group(4))
                w.writerow([j["tier"], j["cid"], j["mode"], j["nz"], j["kg"], j["kext"],
                            j["pw"], EPS_EXT, round(grab(RE["load"], log), 1),
                            round(grab(RE["quant"], log), 1), round(grab(RE["build"], log), 1),
                            round(grab(RE["convert"], log), 1), md, evpk,
                            round(ct, 1), round(base + ct, 1), round(rec, 4)])
                f.flush()
                star = "  <== >=0.8" if rec >= 0.8 else ""
                print(f"[{j['cid']}] md={md} evpK={evpk} recall={rec:.4f} "
                      f"total={base + ct:.1f}s{star}", flush=True)

    print(f"\n[overnight] Done after {(time.time() - t0)/3600:.2f}h -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
