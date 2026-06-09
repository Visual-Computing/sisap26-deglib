#!/usr/bin/env python3
"""
k_ext_sweep.py — Does a lower k_ext make the kg26 winner faster without losing
recall? Build the kg26 graph (nz=608, prune=9, eps=0.002) for several k_ext on
the AVX2 image, sweep max_dist at evpK=50. k_ext only affects BUILD time, so if
recall holds, lower k_ext = faster total.
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HOME", "/home/nico/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, "/opt/sisap26-deglib")
from docker_runner import Task1Runner

OUT = Path("/opt/sisap26-deglib/optuna")
RES = OUT / "container_results"
RES.mkdir(parents=True, exist_ok=True)

K_EXT_LIST = [16, 20, 26, 32, 40, 50]   # low end (cheaper build) + Nico's high end (better graph?)
MD_LIST = "400,500,600,700,800,1000"
EVPK = "50"
NZ = 600
EPS = 0.001
PRUNE = 9   # ~35% of k_graph=26

RE = {k: re.compile(p) for k, p in {
    "load": r"Load Time:\s*([\d.]+)", "quant": r"Quantize Time:\s*([\d.]+)",
    "build": r"Graph Build Time:\s*([\d.]+)", "convert": r"Graph Conversion Time:\s*([\d.]+)",
    "prune": r"Pruning Time:\s*([\d.]+)"}.items()}
RE_COMBO = re.compile(
    r"evpK=(\d+),\s*max_dist=(\d+)\s*has recall\s*([\d.]+)\s*%\s*and time\s*([\d.]+)\s*s")


def grab(rx, log):
    m = rx.search(log)
    return float(m.group(1)) if m else 0.0


runner = Task1Runner(image_tag="sisap26-deglib-cpp:avx2", results_dir=RES, echo_logs=False)
out = OUT / "k_ext_sweep.csv"
results = []
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k_ext", "build_s", "base_s", "max_dist", "evpK", "explore_rerank_s",
                "total_s", "recall"])
    for kext in K_EXT_LIST:
        print(f"\n===== AVX2 kg26 k_ext={kext} (nz={NZ}, eps={EPS}, prune={PRUNE}) | "
              f"md=[{MD_LIST}] evpK={EVPK} =====", flush=True)
        res = runner.run(mode="mode4", size="large", threads=8, non_zeros=NZ, k_graph=26,
                         k_ext=kext, eps_ext=EPS, prune_worst=PRUNE,
                         max_dist=MD_LIST, evp_k=EVPK, timeout_s=4200)
        log = "\n".join(res.raw_logs)
        build = grab(RE["build"], log)
        base = sum(grab(RE[k], log) for k in ("load", "quant", "build", "convert", "prune"))
        for m in RE_COMBO.finditer(log):
            md = int(m.group(2)); rec = float(m.group(3)) / 100.0; ct = float(m.group(4))
            results.append((kext, build, md, rec, base + ct))
            w.writerow([kext, round(build, 1), round(base, 1), md, EVPK,
                        round(ct, 1), round(base + ct, 1), round(rec, 4)])
            f.flush()
            star = "  <== >=0.8" if rec >= 0.8 else ""
            print(f"  k_ext={kext} md={md} recall={rec:.4f} build={build:.0f}s "
                  f"total={base + ct:.1f}s{star}", flush=True)

print("\n=== fastest >=0.8 per k_ext (AVX2) ===", flush=True)
for kext in K_EXT_LIST:
    ok = [r for r in results if r[0] == kext and r[3] >= 0.80]
    if ok:
        b = min(ok, key=lambda r: r[4])
        print(f"  k_ext={kext:>2}: build={b[1]:.0f}s  md={b[2]}  recall={b[3]:.4f}  "
              f"total={b[4]:.1f}s", flush=True)
    else:
        print(f"  k_ext={kext:>2}: never reached 0.8 in this md range", flush=True)
print(f"\nWrote {out}", flush=True)
