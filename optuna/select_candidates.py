#!/usr/bin/env python3
"""
select_candidates.py — From the overnight large sweep, pick the 15 submission
candidates (10 mode4 + 5 mode7) and report the key findings.

Goal = fastest config reaching Recall@15 >= 0.8 on the 6.35M holdout. We bracket
the boundary: select along the (time, recall) front in recall band [0.798, 0.85],
denser at the bottom (fastest winners + small live-shift hedge), thinner toward
0.85 (insurance if the live set is harder).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/opt/sisap26-deglib/optuna")
CSV = OUT / "large_overnight.csv"
# Floor below 0.80: include the fastest sub-0.80 configs as aggressive slots that
# win iff the live set is slightly more forgiving than dev (left/favourable shift).
BAND = (0.788, 0.852)
COLORS = {"mode4": "#1f77b4", "mode7": "#d62728"}


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "tier": r["tier"], "mode": r["mode"], "nz": int(r["non_zeros"]),
                "kg": int(r["k_graph"]), "kext": int(r["k_ext"]), "pw": int(r["prune_worst"]),
                "md": int(r["max_dist"]), "evpK": int(r["evpK"]),
                "build": float(r["build_s"]), "total": float(r["total_s"]),
                "recall": float(r["recall"]),
            })
    return rows


def front(rows):
    """Skyline: minimize total time, maximize recall."""
    s = sorted(rows, key=lambda x: (x["total"], -x["recall"]))
    out, best = [], -1.0
    for r in s:
        if r["recall"] > best + 1e-9:
            out.append(r); best = r["recall"]
    return out


def pick(front_band, targets):
    chosen, used = [], set()
    for t in targets:
        cands = [i for i in range(len(front_band)) if i not in used]
        if not cands:
            break
        bi = min(cands, key=lambda i: abs(front_band[i]["recall"] - t))
        used.add(bi); chosen.append(front_band[bi])
    # supplement to len(targets) if dedup collapsed any
    for i in range(len(front_band)):
        if len(chosen) >= len(targets):
            break
        if i not in used:
            used.add(i); chosen.append(front_band[i])
    return sorted(chosen, key=lambda p: p["recall"])


rows = load()
m4 = [r for r in rows if r["mode"] == "mode4"]
m7 = [r for r in rows if r["mode"] == "mode7"]
f4, f7 = front(m4), front(m7)
b4 = [p for p in f4 if BAND[0] <= p["recall"] <= BAND[1]]
b7 = [p for p in f7 if BAND[0] <= p["recall"] <= BAND[1]]

T4 = [0.790, 0.800, 0.806, 0.812, 0.818, 0.824, 0.830, 0.837, 0.843, 0.848]
T7 = [0.806, 0.816, 0.826, 0.837, 0.849]
c4 = pick(b4, T4)
c7 = pick(b7, T7)


def line(p, mode):
    return (f"  {mode} kg={p['kg']:>2} md={p['md']:>4} evpK={p['evpK']:>3} "
            f"nz={p['nz']:>3}  recall={p['recall']:.4f}  total={p['total']:6.1f}s "
            f"(build {p['build']:.0f}s)")


print("=" * 78)
fastest = min([r for r in rows if r["recall"] >= 0.80], key=lambda r: r["total"])
print(f"Fastest config >=0.80 overall: {fastest['mode']} kg={fastest['kg']} "
      f"md={fastest['md']} evpK={fastest['evpK']} nz={fastest['nz']} -> "
      f"recall={fastest['recall']:.4f} @ {fastest['total']:.1f}s")
print("=" * 78)

print(f"\n--- 10 mode4 candidates (front, recall band {BAND}) ---")
for p in c4:
    print(line(p, "mode4"))
print(f"\n--- 5 mode7 candidates ---")
for p in c7:
    print(line(p, "mode7"))

# evpK effect (Tier C): same mode/kg/md across evpK
print("\n--- evpK effect (Tier C, mode4) ---")
cT = [r for r in rows if r["tier"] == "C"]
for kg in (24, 28, 32):
    for md in (800,):
        grp = sorted([r for r in cT if r["mode"] == "mode4" and r["kg"] == kg and r["md"] == md]
                     + [r for r in rows if r["tier"] == "A" and r["kg"] == kg and r["md"] == md],
                     key=lambda r: r["evpK"])
        if grp:
            s = "  ".join(f"e{r['evpK']}:{r['recall']:.3f}/{r['total']:.0f}s" for r in grp)
            print(f"  kg{kg} md{md}: {s}")

# nz effect (Tier D vs A)
print("\n--- non_zeros effect (mode4, md=800, evpK=50) ---")
for kg in (28, 32):
    grp = sorted([r for r in rows if r["mode"] == "mode4" and r["kg"] == kg and r["md"] == 800
                  and r["evpK"] == 50], key=lambda r: r["nz"])
    if grp:
        s = "  ".join(f"nz{r['nz']}:{r['recall']:.3f}/{r['total']:.0f}s" for r in grp)
        print(f"  kg{kg}: {s}")

# write candidates
with open(OUT / "candidates_15.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["slot", "mode", "k_graph", "max_dist", "evpK", "non_zeros", "k_ext",
                "prune_worst", "eps_ext", "recall_dev_large", "total_s_avx512", "build_s"])
    for i, p in enumerate(c4 + c7, 1):
        w.writerow([i, p["mode"], p["kg"], p["md"], p["evpK"], p["nz"], p["kext"],
                    p["pw"], 0.002, p["recall"], p["total"], p["build"]])

# plot
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.axhspan(BAND[0], BAND[1], color="gold", alpha=0.12, label=f"selection band {BAND}")
for mode, fr, sel in (("mode4", f4, c4), ("mode7", f7, c7)):
    pts = [r for r in rows if r["mode"] == mode]
    ax.scatter([r["total"] for r in pts], [r["recall"] for r in pts],
               s=10, alpha=0.15, color=COLORS[mode])
    ax.plot([p["total"] for p in fr], [p["recall"] for p in fr],
            "-", color=COLORS[mode], lw=1.5, alpha=0.7, label=f"{mode} front")
    ax.scatter([p["total"] for p in sel], [p["recall"] for p in sel],
               s=70, color=COLORS[mode], edgecolor="black", zorder=5,
               label=f"{mode} selected ({len(sel)})")
ax.axhline(0.80, ls="--", color="black", lw=1)
ax.set_xlim(250, 600)
ax.set_ylim(0.74, 0.90)
ax.set_xlabel("Total time (s, AVX-512)")
ax.set_ylabel("Recall@15 (dev large, 6.35M)")
ax.set_title("Large dataset: time/recall front + 15 submission candidates")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "report" / "large_candidates.png", dpi=130)
print(f"\nWrote {OUT/'candidates_15.csv'} and report/large_candidates.png")
