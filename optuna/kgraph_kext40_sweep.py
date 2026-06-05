#!/usr/bin/env python3
"""
kgraph_kext40_sweep.py — Final sweep: does the build-vs-search optimum shift
when we use the better k_ext=40 graph across k_graph (Nico's range 24-32)?
mode4, AVX2, nz=600, eps=0.001, prune=35% of k_graph, evpK=50.

kg26 @ k_ext=40 already measured in k_ext_sweep.csv (same params) — combine
those for the full kg ∈ {24,26,28,32} picture in the report.
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

K_EXT = 40
NZ = 600
EPS = 0.001
MD_LIST = "400,500,600,700,800,1000"
EVPK = "50"
# (k_graph, prune=~35% of k_graph)
CONFIGS = [(24, 8), (28, 10), (32, 11)]

RE = {k: re.compile(p) for k, p in {
    "load": r"Load Time:\s*([\d.]+)", "quant": r"Quantize Time:\s*([\d.]+)",
    "build": r"Graph Build Time:\s*([\d.]+)", "convert": r"Graph Conversion Time:\s*([\d.]+)",
    "prune": r"Pruning Time:\s*([\d.]+)"}.items()}
RE_COMBO = re.compile(
    r"evpK=(\d+),\s*max_dist=(\d+)\s*has recall\s*([\d.]+)\s*%\s*and time\s*([\d.]+)\s*s")


def grab(rx, log):
    m = rx.search(log)
    return float(m.group(1)) if m else 0.0


runner = Task1Runner(image_tag="sisap26-deglib:avx2", results_dir=RES, echo_logs=False)
out = OUT / "kgraph_kext40_sweep.csv"
results = []
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k_graph", "k_ext", "prune_worst", "non_zeros", "build_s", "base_s",
                "max_dist", "evpK", "explore_rerank_s", "total_s", "recall"])
    for kg, pw in CONFIGS:
        print(f"\n===== AVX2 kg={kg} k_ext={K_EXT} (nz={NZ}, eps={EPS}, prune={pw}) | "
              f"md=[{MD_LIST}] evpK={EVPK} =====", flush=True)
        res = runner.run(mode="mode4", size="large", threads=8, non_zeros=NZ, k_graph=kg,
                         k_ext=K_EXT, eps_ext=EPS, prune_worst=pw,
                         max_dist=MD_LIST, evp_k=EVPK, timeout_s=4200)
        log = "\n".join(res.raw_logs)
        build = grab(RE["build"], log)
        base = sum(grab(RE[k], log) for k in ("load", "quant", "build", "convert", "prune"))
        for m in RE_COMBO.finditer(log):
            md = int(m.group(2)); rec = float(m.group(3)) / 100.0; ct = float(m.group(4))
            results.append((kg, build, md, rec, base + ct))
            w.writerow([kg, K_EXT, pw, NZ, round(build, 1), round(base, 1), md, EVPK,
                        round(ct, 1), round(base + ct, 1), round(rec, 4)])
            f.flush()
            star = "  <== >=0.8" if rec >= 0.8 else ""
            print(f"  kg={kg} md={md} recall={rec:.4f} build={build:.0f}s "
                  f"total={base + ct:.1f}s{star}", flush=True)

print("\n=== fastest >=0.8 per k_graph (k_ext=40, AVX2) ===", flush=True)
for kg, _ in CONFIGS:
    ok = [r for r in results if r[0] == kg and r[3] >= 0.80]
    if ok:
        b = min(ok, key=lambda r: r[4])
        print(f"  kg={kg:>2}: build={b[1]:.0f}s  md={b[2]}  recall={b[3]:.4f}  total={b[4]:.1f}s", flush=True)
    else:
        print(f"  kg={kg:>2}: never reached 0.8", flush=True)
print(f"\nWrote {out}", flush=True)
