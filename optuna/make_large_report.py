#!/usr/bin/env python3
"""
make_large_report.py — Build the large-dataset tuning + submission report.

Reads large_overnight.csv (186 configs) and candidates_15.csv, renders an extra
parameter-effects plot, and writes optuna/report/large_report.md. The Pareto +
candidates plot (large_candidates.png) is produced by select_candidates.py.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/opt/sisap26-deglib/optuna")
REP = OUT / "report"
REP.mkdir(parents=True, exist_ok=True)


def load(path, conv):
    with open(path) as f:
        return [conv(r) for r in csv.DictReader(f)]


rows = load(OUT / "large_overnight.csv", lambda r: {
    "tier": r["tier"], "mode": r["mode"], "nz": int(r["non_zeros"]),
    "kg": int(r["k_graph"]), "md": int(r["max_dist"]), "evpK": int(r["evpK"]),
    "build": float(r["build_s"]), "total": float(r["total_s"]), "recall": float(r["recall"]),
})
cands = load(OUT / "candidates_15.csv", lambda r: r)


def front(rs):
    s = sorted(rs, key=lambda x: (x["total"], -x["recall"]))
    out, best = [], -1.0
    for r in s:
        if r["recall"] > best + 1e-9:
            out.append(r); best = r["recall"]
    return out


m4 = [r for r in rows if r["mode"] == "mode4"]
m7 = [r for r in rows if r["mode"] == "mode7"]

# ---- extra plot: evpK effect + non_zeros effect -------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
for kg, col in ((24, "#7fb3d5"), (28, "#2e86c1"), (32, "#1b4f72")):
    pts = sorted([r for r in m4 if r["kg"] == kg and r["md"] == 800], key=lambda r: r["evpK"])
    if pts:
        axA.plot([p["evpK"] for p in pts], [p["recall"] for p in pts], "-o", color=col, label=f"k_graph={kg}")
axA.axhline(0.80, ls="--", color="black", lw=1)
axA.set_xlabel("evpK (rerank pool)"); axA.set_ylabel("Recall@15"); axA.set_title("evpK effect (mode4, max_dist=800)")
axA.legend(); axA.grid(alpha=0.3)
for kg, col in ((28, "#2e86c1"), (32, "#1b4f72")):
    pts = sorted([r for r in m4 if r["kg"] == kg and r["md"] == 800 and r["evpK"] == 50], key=lambda r: r["nz"])
    if pts:
        axB.plot([p["nz"] for p in pts], [p["recall"] for p in pts], "-o", color=col, label=f"k_graph={kg}")
axB.axhline(0.80, ls="--", color="black", lw=1)
axB.set_xlabel("non_zeros (EVP sparsity)"); axB.set_ylabel("Recall@15")
axB.set_title("non_zeros effect (mode4, max_dist=800, evpK=50)")
axB.legend(); axB.grid(alpha=0.3)
fig.suptitle("Large dataset — search & build parameter effects", fontsize=13)
fig.tight_layout(); fig.savefig(REP / "large_param_effects.png", dpi=130); plt.close(fig)

# ---- fastest >=0.8 per k_graph (mode4) ----------------------------------------
per_kg = {}
for r in m4:
    if r["recall"] >= 0.80:
        if r["kg"] not in per_kg or r["total"] < per_kg[r["kg"]]["total"]:
            per_kg[r["kg"]] = r

fastest = min([r for r in rows if r["recall"] >= 0.80], key=lambda r: r["total"])

# ---- markdown -----------------------------------------------------------------
md = []
md.append("# SISAP 2026 deglib — Large-Dataset Tuning & Submission Candidates\n")
md.append("Companion to the small-dataset report. Target: **Wikipedia BGE-M3 large** "
          "(6.35 M vectors, 1024-dim, dot product), self-join k=15, on the AVX-512 VM "
          "(8 vCPU / 24 GB).\n")

md.append("## 0. Objective (challenge rules)\n")
md.append("The challenge runs on a **6.35 M holdout** of the same size/distribution. We submit "
          "**15 configs**; all are run and the **fastest one reaching Recall@15 ≥ 0.8 is reported**. "
          "So we optimise for **speed at the 0.8 boundary**, not maximum recall — anything well "
          "above 0.8 is wasted time.\n")

md.append("## 1. Method\n")
md.append("- **Build-once + search sweep** (deterministic grid, not Optuna): each run builds the "
          "EVP graph once and sweeps `max_dist × evpK` over it. Justified because (a) build dominates "
          "and re-searching build params would rebuild the 6.35 M graph every trial, and (b) the "
          "small report showed build-params translate while only the search budget doesn't.\n")
md.append("- **Carried build-param families** from the small Pareto fronts; refined `k_graph` 22→40 "
          "with a fine `max_dist` grid {400…1400}; bonus tiers swept `evpK ∈ {20,30,50,75,100}` and "
          "`non_zeros ∈ {384,512,640,768}`. `eps_ext` fixed at 0.002 (small report: non-driver of "
          "recall + build-time bomb). `threads=8`, `k_top=15` fixed.\n")
md.append(f"- **186 configurations measured** (~9.4 h). **AVX-512 timings** — recall is "
          "SIMD-independent (final), but the *time* ranking that picks the winner must be "
          "re-measured on the AVX2 target server (see §6).\n")

md.append("## 2. Headline result\n")
md.append(f"**Fastest config ≥ 0.8: mode4, k_graph={fastest['kg']}, max_dist={fastest['md']}, "
          f"evpK={fastest['evpK']}, non_zeros={fastest['nz']} → recall {fastest['recall']:.4f} "
          f"@ {fastest['total']:.0f} s.** First configuration in this project to clear 0.8 on large "
          "comfortably (the repo's prior hand-tuned best was 0.7914).\n")

md.append("## 3. Pareto front & the 15 candidates\n")
md.append("![Large front + candidates](large_candidates.png)\n")
md.append("Both fronts rise steeply then plateau; the **mode4 front sits ~90 s left of mode7 at every "
          "recall level** — mode4 dominates for speed-to-0.8. Selection band is recall ∈ [0.79, 0.85], "
          "denser at the bottom (fastest winners + a sub-0.80 bet) and thinner toward 0.85 (insurance).\n")

md.append("### Build-vs-search trade-off — k_graph=26 is the sweet spot\n")
md.append("Total time = build + search. Lower `k_graph` = cheaper build but needs more `max_dist` to "
          "reach 0.8; higher `k_graph` = pricier build, less search. The fastest ≥0.8 per degree:\n")
md.append("| k_graph | build (s) | fastest ≥0.8 (max_dist) | recall | total (s) |\n|--:|--:|--:|--:|--:|")
for kg in sorted(per_kg):
    r = per_kg[kg]
    md.append(f"| {kg} | {r['build']:.0f} | {r['md']} | {r['recall']:.4f} | {r['total']:.0f} |")
md.append("")
md.append("`k_graph ≤ 22` can't reach 0.8 at all (recall ceiling too low — too sparse); `k_graph=26` "
          "minimises total time at the boundary; `k_graph ≥ 32` only pays off above ~0.84.\n")

md.append("## 4. Parameter findings\n")
md.append("![Large parameter effects](large_param_effects.png)\n")
md.append("- **evpK = 50 is the sweet spot.** evpK=20 craters recall (0.70–0.73 — pool too small); "
          "evpK=30 just reaches ~0.80; evpK=75/100 add only ~+0.01 for +12–27 s. evpK=30 is a "
          "faster-but-risky option for the aggressive slots.\n")
md.append("- **`non_zeros` translated from small.** Recall peaks at the carried-over values "
          "(~512–608) and *drops* at 384 and 768 — confirming the small-optimal quantisation "
          "sparsity is also best at scale.\n")
md.append("- **`k_graph` is the dominant build-cost driver** and the key build knob; the boundary "
          "optimum is 26.\n")

md.append("## 5. small → large translation verdict\n")
md.append("- **Build-graph params translate.** `non_zeros` optimum is unchanged; the `k_graph` "
          "recall ordering holds (more degree → more recall at both scales).\n")
md.append("- **`max_dist` does NOT translate 1:1.** The same config loses ~0.10–0.14 recall going "
          "200 K → 6.35 M, so the budget to reach a given recall grows ~2–3× (small cleared 0.8 at "
          "`max_dist`≈200–400; large needs ≈600 at k_graph=26). This was the key reason to re-tune "
          "only the search budget at scale.\n")
md.append("- **mode4 > mode7 holds at both scales.**\n")

md.append("## 6. The 15 submission candidates\n")
md.append("Recall-safety ladder (two-sided hedge against the dev→live recall shift, whose direction "
          "is unknown): one sub-0.80 *aggressive* slot (wins if live ≥ dev), the fastest safe crosser, "
          "and a tail up to ~0.85 (insurance if live is harder). 10 mode4 + 5 mode7.\n")
md.append("| slot | mode | k_graph | max_dist | evpK | non_zeros | recall (dev large) | total (s, AVX-512) |\n"
          "|--:|--|--:|--:|--:|--:|--:|--:|")
for c in cands:
    md.append(f"| {c['slot']} | {c['mode']} | {c['k_graph']} | {c['max_dist']} | {c['evpK']} | "
              f"{c['non_zeros']} | {float(c['recall_dev_large']):.4f} | {float(c['total_s_avx512']):.0f} |")
md.append("")
md.append("All share `eps_ext=0.002`, `k_top=15`, `threads=8`. mode4 is the win bet; the 5 mode7 are "
          "a higher-recall-per-config hedge (slower, but more margin if the live set is harder).\n")

md.append("## 7. Caveats & next steps\n")
md.append("- **AVX-512 timings only.** The recall column is final (SIMD-independent), but the "
          "evaluation server is **AVX2-only**, so the time ranking — and possibly the mode4/mode7 "
          "order — can shift. **Re-time these 15 on an AVX2 build** (`docker build --build-arg "
          "FORCE_AVX2=ON`) before final submission; that run decides the winner.\n")
md.append("- The mode4 ladder is `k_graph=26`-heavy (it *is* the front across 0.79–0.84); 1–2 slots "
          "could be swapped for `k_graph=24/28` build-diversity if hedging against degree-specific "
          "behaviour on the live set.\n")

(REP / "large_report.md").write_text("\n".join(md))
print(f"Wrote {REP/'large_report.md'} and {REP/'large_param_effects.png'}")
