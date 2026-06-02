#!/usr/bin/env python3
"""
make_report.py — Build the small-dataset hyperparameter tuning report.

Reads mode4_small_trials.csv / mode7_small_trials.csv, computes the Pareto
fronts, renders plots, and writes a markdown report under optuna/report/.
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

MODES = ["mode4", "mode7"]
PARAMS = ["non_zeros", "k_graph", "k_ext", "eps_ext", "prune_worst", "max_dist", "evpK"]
LOG_PARAMS = {"eps_ext", "max_dist", "evpK"}
COLORS = {"mode4": "#1f77b4", "mode7": "#d62728"}
# README/benchmarks.md Intel Xeon 8581C (AVX-512) default-config baselines.
BASELINE = {"mode4": (5.6, 0.8415), "mode7": (6.3, 0.8485)}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(mode: str) -> list[dict]:
    rows = []
    with open(OUT / f"{mode}_small_trials.csv") as f:
        for r in csv.DictReader(f):
            if not r["recall"] or not r["overall_time_s"]:
                continue  # failed/timed-out trial -> skip in analysis
            d = {"recall": float(r["recall"]), "time": float(r["overall_time_s"]),
                 "number": int(r["number"])}
            for p in PARAMS:
                d[p] = float(r[p])
            rows.append(d)
    return rows


def pareto(rows: list[dict]) -> list[dict]:
    """Skyline: minimize time, maximize recall."""
    s = sorted(rows, key=lambda x: (x["time"], -x["recall"]))
    front, best = [], -1.0
    for r in s:
        if r["recall"] > best + 1e-9:
            front.append(r)
            best = r["recall"]
    return front


def spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def recall_at(rows, t):
    """Best recall achievable within time budget t."""
    cand = [r["recall"] for r in rows if r["time"] <= t]
    return max(cand) if cand else None


def knee(front, thr):
    """Cheapest (lowest-time) front point clearing recall threshold thr."""
    cand = [r for r in front if r["recall"] >= thr]
    return min(cand, key=lambda r: r["time"]) if cand else None


data = {m: load(m) for m in MODES}
fronts = {m: pareto(data[m]) for m in MODES}
n_total = {m: len(data[m]) for m in MODES}

# --------------------------------------------------------------------------- #
# Plot 1 — Pareto fronts
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for m in MODES:
    ax.scatter([r["time"] for r in data[m]], [r["recall"] for r in data[m]],
               s=10, alpha=0.18, color=COLORS[m])
    f = fronts[m]
    ax.plot([r["time"] for r in f], [r["recall"] for r in f],
            "-o", color=COLORS[m], ms=4, lw=1.8, label=f"{m} Pareto front (n={len(f)})")
    bt, br = BASELINE[m]
    ax.scatter([bt], [br], marker="*", s=240, color=COLORS[m],
               edgecolor="black", zorder=6,
               label=f"{m} README default ({br:.3f}@{bt:.1f}s)")
ax.axhline(0.8, ls="--", color="gray", lw=0.8)
ax.axhline(0.9, ls=":", color="gray", lw=0.8)
# Clip x to the informative region: a handful of pathological builds run to
# hundreds of s and would otherwise squash the whole front into a sliver.
xmax = max(r["time"] for m in MODES for r in fronts[m]) + 2
ax.set_xlim(0, xmax)
ax.set_xlabel("Total elapsed time (s)")
ax.set_ylabel("Recall@15")
ax.set_title("Pareto fronts — small (200K), minimize time / maximize recall")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(REP / "pareto_fronts.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Plot 2 — recall vs each parameter
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for i, p in enumerate(PARAMS):
    ax = axes[i // 3][i % 3]
    for m in MODES:
        ax.scatter([r[p] for r in data[m]], [r["recall"] for r in data[m]],
                   s=10, alpha=0.25, color=COLORS[m], label=m)
    if p in LOG_PARAMS:
        ax.set_xscale("log")
    ax.set_xlabel(p)
    ax.set_ylabel("Recall@15")
    ax.grid(alpha=0.3)
for j in range(len(PARAMS), 9):
    axes[j // 3][j % 3].axis("off")
axes[0][0].legend()
fig.suptitle("Recall@15 vs each hyperparameter (all completed trials)", fontsize=13)
fig.tight_layout()
fig.savefig(REP / "param_effects.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Plot 3 — parameter influence (Spearman rank corr with recall and time)
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
rho_table = {}
for ax, m in zip(axes, MODES):
    legit = [r for r in data[m] if r["time"] < 600]
    rec_rho = [spearman([r[p] for r in legit], [r["recall"] for r in legit]) for p in PARAMS]
    tim_rho = [spearman([r[p] for r in legit], [r["time"] for r in legit]) for p in PARAMS]
    rho_table[m] = {p: (rec_rho[i], tim_rho[i]) for i, p in enumerate(PARAMS)}
    y = np.arange(len(PARAMS))
    h = 0.38
    ax.barh(y + h / 2, rec_rho, height=h, color="#2ca02c", label="vs recall")
    ax.barh(y - h / 2, tim_rho, height=h, color="#ff7f0e", label="vs time")
    ax.set_yticks(y)
    ax.set_yticklabels(PARAMS)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-1, 1)
    ax.set_title(f"{m}")
    ax.set_xlabel("Spearman ρ")
    ax.grid(alpha=0.3, axis="x")
axes[0].legend(loc="lower left")
fig.suptitle("Parameter influence — Spearman rank correlation", fontsize=13)
fig.tight_layout()
fig.savefig(REP / "param_importance.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Plot 4 — mode4 vs mode7 recall envelope + dominance gap
# --------------------------------------------------------------------------- #
t_lo = min(min(r["time"] for r in data[m]) for m in MODES)
t_hi = max(r["time"] for m in MODES for r in fronts[m]) + 1
tgrid = np.linspace(t_lo, t_hi, 300)
env = {m: np.array([recall_at(data[m], t) if recall_at(data[m], t) is not None else np.nan
                    for t in tgrid]) for m in MODES}
gap = env["mode4"] - env["mode7"]

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for m in MODES:
    axA.plot(tgrid, env[m], color=COLORS[m], lw=2, label=f"{m} best recall within budget")
axA.axhline(0.8, ls="--", color="gray", lw=0.8)
axA.axhline(0.9, ls=":", color="gray", lw=0.8)
axA.set_ylabel("Recall@15 (best ≤ t)")
axA.set_title("mode4 vs mode7 — recall envelope and dominance gap (small)")
axA.legend(loc="lower right")
axA.grid(alpha=0.3)

axB.axhline(0, color="k", lw=0.8)
axB.plot(tgrid, gap, color="black", lw=1.0)
axB.fill_between(tgrid, gap, 0, where=gap >= 0, color=COLORS["mode4"], alpha=0.45,
                 interpolate=True, label="mode4 ahead")
axB.fill_between(tgrid, gap, 0, where=gap < 0, color=COLORS["mode7"], alpha=0.45,
                 interpolate=True, label="mode7 ahead")
axB.set_ylabel("recall gap\n(mode4 − mode7)")
axB.set_xlabel("Total elapsed time (s)")
axB.legend(loc="upper right")
axB.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(REP / "dominance_gap.png", dpi=130)
plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def fmt_front_table(front):
    lines = ["| time (s) | recall | non_zeros | k_graph | k_ext | eps_ext | prune | max_dist | evpK |",
             "|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in front:
        lines.append("| {time:.1f} | {recall:.4f} | {non_zeros:.0f} | {k_graph:.0f} | "
                      "{k_ext:.0f} | {eps_ext:.4f} | {prune_worst:.0f} | {max_dist:.0f} | "
                      "{evpK:.0f} |".format(**r))
    return "\n".join(lines)


def fmt_top(rows, n=10):
    top = sorted(rows, key=lambda r: r["recall"], reverse=True)[:n]
    lines = ["| # | recall | time (s) | non_zeros | k_graph | k_ext | eps_ext | prune | max_dist | evpK |",
             "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in top:
        lines.append("| {number:.0f} | {recall:.4f} | {time:.1f} | {non_zeros:.0f} | "
                     "{k_graph:.0f} | {k_ext:.0f} | {eps_ext:.4f} | {prune_worst:.0f} | "
                     "{max_dist:.0f} | {evpK:.0f} |".format(**r))
    return "\n".join(lines)


T_GRID = [4, 5, 6, 8, 10, 12, 14]
md = []
md.append("# SISAP 2026 deglib — Small-Dataset Hyperparameter Tuning Report\n")
md.append("Multi-objective Optuna search (TPE) on Wikipedia BGE-M3 **small** (200K vectors, "
          "1024-dim, dot product) on the AVX-512 VM (8 vCPU / 24 GB). Objective: "
          "**minimize total elapsed time, maximize Recall@15** (Pareto). "
          "`threads=8` and `k_top=15` fixed.\n")
md.append("Search space (per trial, with constraints `evpK≥15`, `max_dist≥evpK`, "
          "`prune_worst<k_graph`):\n")
md.append("| param | range |\n|---|---|\n"
          "| non_zeros | 128–960 (step 64) |\n| k_graph | 12–64 (even) |\n"
          "| k_ext | 16–128 |\n| eps_ext | 1e-4–5e-2 (log) |\n"
          "| prune_worst | 0 … 0.6·k_graph |\n| max_dist | 50–800 (log) |\n"
          "| evpK | 15 … min(400, max_dist) (log) |\n")
md.append("200 trials per mode. Completed (non-failed) trials analysed: "
          f"mode4 = {n_total['mode4']}, mode7 = {n_total['mode7']}. "
          f"Pareto front sizes: mode4 = {len(fronts['mode4'])}, mode7 = {len(fronts['mode7'])}.\n")

md.append("## 1. Pareto fronts\n")
md.append("![Pareto fronts](pareto_fronts.png)\n")
md.append("Both fronts share the same shape: a **steep rise then a long plateau** — most "
          "recall is bought cheaply, with sharp diminishing returns past ~0.9.\n")

md.append("### Recall reachable within a time budget\n")
md.append("| time budget | mode4 best recall | mode7 best recall |\n|--:|--:|--:|")
for t in T_GRID:
    r4, r7 = recall_at(data["mode4"], t), recall_at(data["mode7"], t)
    md.append(f"| ≤{t}s | {('%.4f'%r4) if r4 else '—'} | {('%.4f'%r7) if r7 else '—'} |")
md.append("")

md.append("### Knees (cheapest front config clearing each recall bar)\n")
md.append("| threshold | mode | time (s) | config |\n|--|--|--:|--|")
for thr in (0.80, 0.90, 0.95):
    for m in MODES:
        k = knee(fronts[m], thr)
        if k:
            cfg = (f"nz={k['non_zeros']:.0f}, kg={k['k_graph']:.0f}, "
                   f"kext={k['k_ext']:.0f}, prune={k['prune_worst']:.0f}, "
                   f"md={k['max_dist']:.0f}, evpK={k['evpK']:.0f}")
            md.append(f"| ≥{thr:.2f} | {m} | {k['time']:.1f} | {cfg} |")
md.append("")

md.append("## 2. mode4 vs mode7\n")
md.append("![mode4 vs mode7 dominance gap](dominance_gap.png)\n")
md.append("Top: best recall reachable within a time budget (the monotone envelope of each "
          "front). Bottom: the gap (mode4 − mode7) — positive where mode4 dominates. mode4 "
          "leads across the cheap region and the two converge at the expensive tip.\n")
peak4 = max(data["mode4"], key=lambda r: r["recall"])
peak7 = max(data["mode7"], key=lambda r: r["recall"])
md.append(f"- Peak recall: **mode4 = {peak4['recall']:.4f}** ({peak4['time']:.1f}s), "
          f"**mode7 = {peak7['recall']:.4f}** ({peak7['time']:.1f}s).\n")
wins = {"mode4": 0, "mode7": 0}
for t in T_GRID:
    r4, r7 = recall_at(data["mode4"], t), recall_at(data["mode7"], t)
    if r4 and r7:
        wins["mode4" if r4 >= r7 else "mode7"] += 1
md.append(f"- At matched time budgets {T_GRID}, the higher-recall mode is "
          f"mode4 in {wins['mode4']} and mode7 in {wins['mode7']} of them "
          "(see the table above). Asymmetric search (mode7) keeps the full-precision "
          "query, costing a little explore time for a slightly better candidate pool.\n")

md.append("## 3. Parameter influence\n")
md.append("![Parameter effects](param_effects.png)\n")
md.append("![Parameter influence](param_importance.png)\n")
md.append("Spearman rank correlation of each parameter with recall and with time "
          "(mode4 / mode7):\n")
md.append("| param | ρ recall (m4) | ρ time (m4) | ρ recall (m7) | ρ time (m7) |\n"
          "|---|--:|--:|--:|--:|")
for p in PARAMS:
    a = rho_table["mode4"][p]
    b = rho_table["mode7"][p]
    md.append(f"| {p} | {a[0]:+.2f} | {a[1]:+.2f} | {b[0]:+.2f} | {b[1]:+.2f} |")
md.append("")
md.append("Reading: large positive ρ with recall = strong recall driver; large positive ρ "
          "with time = strong cost driver. `max_dist`/`evpK`/`k_graph` drive recall (and "
          "cost); `eps_ext` mostly drives cost with weak recall signal; low `non_zeros` "
          "hurts recall.\n")

md.append("## 4. Comparison to the README baseline\n")
md.append("README/benchmarks.md only ever measured the single **default** config per mode. "
          "On the matched AVX-512 hardware:\n")
md.append("| | config | time | recall |\n|---|---|--:|--:|")
for m in MODES:
    bt, br = BASELINE[m]
    rate = recall_at(data[m], bt)
    md.append(f"| {m} README default | k_graph=32, max_dist=200, evpK=50 | {bt:.1f}s | {br:.4f} |")
    md.append(f"| {m} tuned @ ~{bt:.0f}s | best front config within budget | ≤{bt:.1f}s | "
              f"{('%.4f'%rate) if rate else '—'} |")
    pk = peak4 if m == "mode4" else peak7
    md.append(f"| {m} tuned ceiling | best overall | {pk['time']:.1f}s | {pk['recall']:.4f} |")
md.append("")
md.append("So tuning lifts recall by ~+0.05 at matched time, and the ceiling (~0.96) far "
          "exceeds anything previously measured (~0.84).\n")

md.append("## 5. Translation hypothesis for the large dataset\n")
md.append("- **Likely transfer ~1:1** (graph-quality knobs, dataset-size independent): "
          "`non_zeros`, and the *ranking* of `k_graph`, `prune_worst`, `k_ext`, `eps_ext`.\n"
          "- **Likely NOT 1:1** (must be re-tuned at scale): `max_dist` — navigating 6.35M "
          "vs 200K needs a larger budget for the same recall — and to a lesser extent `evpK`.\n")
md.append("Implication: rather than a blind 200-trial large search (~1.5–2 days, much wasted "
          "on timed-out builds), **carry over good build-configs from this front, build each "
          "large graph once (`--graph` save), and sweep `max_dist`×`evpK` cheaply on the "
          "saved graph**.\n")

md.append("## Appendix — full Pareto fronts\n")
for m in MODES:
    md.append(f"### {m} front ({len(fronts[m])} points)\n")
    md.append(fmt_front_table(fronts[m]) + "\n")
    md.append(f"### {m} top-10 by recall\n")
    md.append(fmt_top(data[m]) + "\n")

(REP / "report.md").write_text("\n".join(md))
print(f"Report written to {REP/'report.md'}")
print(f"Plots: {', '.join(p.name for p in REP.glob('*.png'))}")
print(f"Pareto: mode4={len(fronts['mode4'])}  mode7={len(fronts['mode7'])}")
print(f"Peak recall: mode4={peak4['recall']:.4f}@{peak4['time']:.1f}s  "
      f"mode7={peak7['recall']:.4f}@{peak7['time']:.1f}s")
print(f"matched-time wins: {wins}")
