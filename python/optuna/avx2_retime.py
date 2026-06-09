#!/usr/bin/env python3
"""
avx2_retime.py — Re-time the 15 submission candidates on the AVX2 image
(matches the AVX2-only evaluation server). Recall is ~SIMD-independent; the
timings (and possibly the ranking) are what change. Groups candidates by
build-config so each graph builds once and sweeps its needed max_dist x evpK.

Reads optuna/candidates_15.csv, writes optuna/candidates_15_avx2.csv.
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
RE = {k: re.compile(p) for k, p in {
    "load": r"Load Time:\s*([\d.]+)", "quant": r"Quantize Time:\s*([\d.]+)",
    "build": r"Graph Build Time:\s*([\d.]+)", "convert": r"Graph Conversion Time:\s*([\d.]+)",
    "prune": r"Pruning Time:\s*([\d.]+)"}.items()}
RE_COMBO = re.compile(
    r"evpK=(\d+),\s*max_dist=(\d+)\s*has recall\s*([\d.]+)\s*%\s*and time\s*([\d.]+)\s*s")


def grab(rx, log):
    m = rx.search(log)
    return float(m.group(1)) if m else 0.0


# Load candidates and group by build-config
cands = []
with open(OUT / "candidates_15.csv") as f:
    for r in csv.DictReader(f):
        cands.append(r)

groups: dict[tuple, list] = {}
for c in cands:
    key = (c["mode"], int(c["k_graph"]), int(c["non_zeros"]), int(c["k_ext"]), int(c["prune_worst"]))
    groups.setdefault(key, []).append(c)

runner = Task1Runner(image_tag="sisap26-deglib-cpp:avx2", results_dir=RES, echo_logs=False)
avx2: dict[tuple, tuple] = {}   # (mode,kg,md,evpK) -> (recall, total_s)

for (mode, kg, nz, kext, pw), members in groups.items():
    md_list = ",".join(str(x) for x in sorted({int(m["max_dist"]) for m in members}))
    evpk_list = ",".join(str(x) for x in sorted({int(m["evpK"]) for m in members}))
    print(f"\n===== AVX2 {mode} kg={kg} nz={nz} | md=[{md_list}] evpK=[{evpk_list}] =====", flush=True)
    res = runner.run(mode=mode, size="large", threads=8, non_zeros=nz, k_graph=kg,
                     k_ext=kext, eps_ext=0.002, prune_worst=pw,
                     max_dist=md_list, evp_k=evpk_list, timeout_s=4200)
    log = "\n".join(res.raw_logs)
    base = sum(grab(RE[k], log) for k in ("load", "quant", "build", "convert", "prune"))
    for m in RE_COMBO.finditer(log):
        evpk, md = int(m.group(1)), int(m.group(2))
        rec, ct = float(m.group(3)) / 100.0, float(m.group(4))
        avx2[(mode, kg, md, evpk)] = (rec, base + ct)
        print(f"  md={md} evpK={evpk} recall={rec:.4f} total={base + ct:.1f}s", flush=True)

# Write merged comparison
out = OUT / "candidates_15_avx2.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["slot", "mode", "k_graph", "k_ext", "prune_worst", "non_zeros", "max_dist", "evpK",
                "recall_avx512", "total_avx512", "recall_avx2", "total_avx2", "avx2/avx512"])
    for c in cands:
        key = (c["mode"], int(c["k_graph"]), int(c["max_dist"]), int(c["evpK"]))
        rec2, tot2 = avx2.get(key, (None, None))
        t512 = float(c["total_s_avx512"])
        ratio = (tot2 / t512) if tot2 else None
        w.writerow([c["slot"], c["mode"], c["k_graph"], c["k_ext"], c["prune_worst"],
                    c["non_zeros"], c["max_dist"], c["evpK"],
                    c["recall_dev_large"], round(t512, 1),
                    round(rec2, 4) if rec2 is not None else "",
                    round(tot2, 1) if tot2 is not None else "",
                    round(ratio, 3) if ratio else ""])

print(f"\nWrote {out}")
print("\n=== AVX2 ranking (>=0.8, fastest first) ===", flush=True)
ranked = sorted([(c, *avx2.get((c["mode"], int(c["k_graph"]), int(c["max_dist"]), int(c["evpK"])), (None, None)))
                 for c in cands], key=lambda x: (x[2] if x[2] else 9e9))
for c, rec2, tot2 in ranked:
    if rec2 is None:
        continue
    star = "  <== >=0.8" if rec2 >= 0.8 else ""
    print(f"  {c['mode']} kg={c['k_graph']:>2} md={c['max_dist']:>4} evpK={c['evpK']:>3}  "
          f"recall={rec2:.4f}  avx2={tot2:6.1f}s  (avx512 {float(c['total_s_avx512']):.0f}s){star}", flush=True)
