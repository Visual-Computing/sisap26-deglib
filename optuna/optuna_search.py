#!/usr/bin/env python3
"""
optuna_search.py — Hyperparameter search for SISAP 2026 deglib (mode4 & mode7).

Two phases
----------
search : Multi-objective Pareto study on the SMALL dataset.
         Objectives = (minimize overall_time_s, maximize recall@15).
         One trial = one fully-specified config = one container run.

large  : Rerun the top-N-by-recall configs (+ a few spread points across the
         small Pareto front) on the LARGE dataset, to answer:
           (1) can we reach 0.8 recall on large in the current state?
           (2) do hyperparameters translate from small -> large?

Everything (sqlite study DB + CSV exports) is written under OUT_DIR so the
study is resumable and inspectable.

Usage
-----
    python optuna_search.py search --mode mode4 --trials 200
    python optuna_search.py search --mode mode7 --trials 200
    python optuna_search.py large  --mode mode4 --top 10 --spread 5
    python optuna_search.py report --mode mode4
"""
from __future__ import annotations

# --- HF cache must be configured BEFORE docker_runner imports huggingface_hub ---
import os
os.environ.setdefault("HF_HOME", "/home/nico/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # datasets are already cached

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, "/opt/sisap26-deglib")             # make docker_runner importable

import optuna
from optuna.trial import TrialState

from docker_runner import Task1Runner

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path("/opt/sisap26-deglib/optuna")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{OUT_DIR / 'study.db'}"

MODES = ("mode4", "mode7")

# optuna param name -> Task1Runner.run() kwarg name
RUNNER_KW = {
    "non_zeros":   "non_zeros",
    "k_graph":     "k_graph",
    "k_ext":       "k_ext",
    "eps_ext":     "eps_ext",
    "prune_worst": "prune_worst",
    "max_dist":    "max_dist",
    "evpK":        "evp_k",
}
PARAM_ORDER = ["non_zeros", "k_graph", "k_ext", "eps_ext",
               "prune_worst", "max_dist", "evpK"]

# Sentinel objective values for a failed/invalid trial (fully dominated point).
FAIL_TIME = 1.0e9
FAIL_RECALL = 0.0

# Per-run wall-clock ceilings. Legit small runs are <15 s; pathological build
# configs (low non_zeros + high k_graph + high eps_ext) can run for many
# minutes, so kill them and record a dominated point. Large runs are ~400-500 s
# legit, so the ceiling is generous.
SMALL_TIMEOUT_S = 180.0
LARGE_TIMEOUT_S = 1200.0


def study_name(mode: str) -> str:
    return f"{mode}_small"


def make_runner() -> Task1Runner:
    return Task1Runner(
        image_tag="sisap26-deglib",
        results_dir=OUT_DIR / "container_results",
        echo_logs=False,
    )


def to_kwargs(params: dict) -> dict:
    """Map an optuna params dict to Task1Runner.run() keyword arguments."""
    return {RUNNER_KW[k]: v for k, v in params.items() if k in RUNNER_KW}


def norm_recall(r: float | None) -> float | None:
    """The binary reports Recall@15 as a percent (e.g. 84.21). Normalise to 0-1
    so the >= 0.8 target and all stored values live on a single fraction scale."""
    if r is None:
        return None
    return r / 100.0 if r > 1.0 else r


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict:
    """Broad search space with the agreed coupling constraints baked in."""
    # --- build-graph params -------------------------------------------------
    # non_zeros must be strictly < dim (=1024); the binary aborts otherwise.
    # Top = 1024-64 = 960; lowest grid point 128.
    non_zeros = trial.suggest_int("non_zeros", 128, 960, step=64)
    k_graph   = trial.suggest_int("k_graph", 12, 64, step=2)        # even degree
    k_ext     = trial.suggest_int("k_ext", 16, 128)
    eps_ext   = trial.suggest_float("eps_ext", 1e-4, 5e-2, log=True)
    # prune_worst sampled relative to degree (must stay < k_graph)
    prune_hi  = max(0, int(0.6 * k_graph))
    prune_worst = trial.suggest_int("prune_worst", 0, prune_hi)

    # --- search / budget params --------------------------------------------
    max_dist = trial.suggest_int("max_dist", 50, 800, log=True)
    evpk_hi  = min(400, max_dist)                                    # evpK <= max_dist
    evp_k    = trial.suggest_int("evpK", 15, evpk_hi, log=True)      # evpK >= k_top(15)

    return {
        "non_zeros": non_zeros,
        "k_graph": k_graph,
        "k_ext": k_ext,
        "eps_ext": eps_ext,
        "prune_worst": prune_worst,
        "max_dist": max_dist,
        "evpK": evp_k,
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(mode: str, runner: Task1Runner):
    def objective(trial: optuna.Trial):
        params = suggest_params(trial)
        res = runner.run(mode=mode, size="small", threads=8,
                         timeout_s=SMALL_TIMEOUT_S, **to_kwargs(params))

        recall = norm_recall(res.best_recall)
        t = res.overall_time_s
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("overall_time_s", t)
        trial.set_user_attr("exit_code", res.exit_code)

        ok = res.succeeded and recall is not None and t is not None
        print(
            f"[{mode}] trial {trial.number:>4} "
            f"recall={recall if recall is not None else 'NA'} "
            f"time={t if t is not None else 'NA'}s exit={res.exit_code} "
            f"params={params}",
            flush=True,
        )
        if not ok:
            trial.set_user_attr("error_tail", res.raw_logs[-15:])
            return FAIL_TIME, FAIL_RECALL
        return float(t), float(recall)

    return objective


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def trials_csv_path(mode: str) -> Path:
    return OUT_DIR / f"{mode}_small_trials.csv"


def make_csv_callback(mode: str):
    path = trials_csv_path(mode)

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        write_header = not path.exists()
        ua, pp = trial.user_attrs, trial.params
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    ["number", "state", "recall", "overall_time_s", "exit_code"]
                    + PARAM_ORDER
                )
            w.writerow(
                [trial.number, trial.state.name,
                 ua.get("recall"), ua.get("overall_time_s"), ua.get("exit_code")]
                + [pp.get(k) for k in PARAM_ORDER]
            )

    return cb


# ---------------------------------------------------------------------------
# Phase 1: search on small
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    mode = args.mode
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=args.seed)
    study = optuna.create_study(
        study_name=study_name(mode),
        storage=STORAGE,
        directions=["minimize", "maximize"],   # (time, recall)
        sampler=sampler,
        load_if_exists=True,
    )
    runner = make_runner()
    done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    print(f"[{mode}] resuming with {done} completed trials; "
          f"targeting {args.trials} new trials.", flush=True)
    study.optimize(
        make_objective(mode, runner),
        n_trials=args.trials,
        callbacks=[make_csv_callback(mode)],
        gc_after_trial=True,
    )
    print(f"[{mode}] search done. Pareto front size = {len(study.best_trials)}.",
          flush=True)


# ---------------------------------------------------------------------------
# Phase 2: rerun selected configs on large
# ---------------------------------------------------------------------------

def _completed(study: optuna.Study) -> list:
    return [t for t in study.trials
            if t.state == TrialState.COMPLETE
            and t.user_attrs.get("recall") is not None]


def _select_configs(study: optuna.Study, top: int, spread: int) -> list:
    """Top-N by small recall + `spread` evenly-spaced Pareto-front points."""
    completed = _completed(study)
    by_recall = sorted(completed, key=lambda t: t.user_attrs["recall"], reverse=True)

    selected, seen = [], set()

    def key(t):
        return tuple(t.params.get(k) for k in PARAM_ORDER)

    for t in by_recall[:top]:
        k = key(t)
        if k not in seen:
            seen.add(k)
            selected.append(("top_recall", t))

    if spread > 0:
        front = sorted(study.best_trials, key=lambda t: t.user_attrs.get("recall", 0))
        if front:
            # evenly spaced indices across the front (by recall)
            idxs = sorted({round(i * (len(front) - 1) / max(1, spread - 1))
                           for i in range(spread)})
            for i in idxs:
                t = front[i]
                k = key(t)
                if k not in seen:
                    seen.add(k)
                    selected.append(("spread", t))
    return selected


def cmd_large(args: argparse.Namespace) -> None:
    mode = args.mode
    study = optuna.load_study(study_name=study_name(mode), storage=STORAGE)
    selected = _select_configs(study, args.top, args.spread)
    if not selected:
        print(f"[{mode}] no completed trials to rerun.", file=sys.stderr)
        return

    runner = make_runner()
    out_path = OUT_DIR / f"{mode}_large_rerun.csv"
    rows = []
    print(f"[{mode}] rerunning {len(selected)} configs on LARGE …", flush=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["source", "small_trial", "small_recall", "small_time_s",
             "large_recall", "large_time_s", "large_exit"]
            + PARAM_ORDER
        )
        for source, t in selected:
            kwargs = to_kwargs(t.params)
            res = runner.run(mode=mode, size="large", threads=8,
                             timeout_s=LARGE_TIMEOUT_S, **kwargs)
            large_recall = norm_recall(res.best_recall)
            row = {
                "source": source,
                "small_trial": t.number,
                "small_recall": t.user_attrs.get("recall"),
                "small_time_s": t.user_attrs.get("overall_time_s"),
                "large_recall": large_recall,
                "large_time_s": res.overall_time_s,
                "large_exit": res.exit_code,
                **{k: t.params.get(k) for k in PARAM_ORDER},
            }
            rows.append(row)
            w.writerow([row["source"], row["small_trial"], row["small_recall"],
                        row["small_time_s"], row["large_recall"],
                        row["large_time_s"], row["large_exit"]]
                       + [row[k] for k in PARAM_ORDER])
            f.flush()
            star = " <== >= 0.8" if (large_recall or 0) >= 0.8 else ""
            print(f"[{mode}] trial {t.number} ({source}): "
                  f"small={row['small_recall']} -> large={large_recall} "
                  f"time={res.overall_time_s}s exit={res.exit_code}{star}",
                  flush=True)

    _print_translation(mode, rows)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, no scipy."""
    n = len(xs)
    if n < 2:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _print_translation(mode: str, rows: list[dict]) -> None:
    valid = [r for r in rows
             if r["large_recall"] is not None and r["small_recall"] is not None]
    print("\n" + "=" * 64)
    print(f"  {mode}: small -> large translation")
    print("=" * 64)
    n_hit = sum(1 for r in valid if (r["large_recall"] or 0) >= 0.8)
    print(f"  configs rerun on large : {len(valid)}")
    print(f"  reached >= 0.8 on large: {n_hit}")
    if valid:
        best = max(valid, key=lambda r: r["large_recall"])
        print(f"  best large recall      : {best['large_recall']:.4f} "
              f"(trial {best['small_trial']}, small={best['small_recall']:.4f})")
        rho = _spearman([r["small_recall"] for r in valid],
                        [r["large_recall"] for r in valid])
        if rho is not None:
            print(f"  Spearman(small,large)  : {rho:+.3f} "
                  f"(rank translation of recall)")
    print("=" * 64, flush=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    study = optuna.load_study(study_name=study_name(args.mode), storage=STORAGE)
    completed = _completed(study)
    print(f"[{args.mode}] completed trials: {len(completed)}")
    print(f"[{args.mode}] Pareto-front size: {len(study.best_trials)}")
    top = sorted(completed, key=lambda t: t.user_attrs["recall"], reverse=True)[:args.top]
    print(f"\nTop {len(top)} by small recall:")
    print(f"  {'#':>5} {'recall':>8} {'time_s':>9}  params")
    for t in top:
        print(f"  {t.number:>5} {t.user_attrs['recall']:>8.4f} "
              f"{t.user_attrs['overall_time_s']:>9.2f}  "
              f"{ {k: t.params.get(k) for k in PARAM_ORDER} }")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="run the Pareto search on the small dataset")
    s.add_argument("--mode", choices=MODES, required=True)
    s.add_argument("--trials", type=int, default=200)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_search)

    l = sub.add_parser("large", help="rerun selected configs on the large dataset")
    l.add_argument("--mode", choices=MODES, required=True)
    l.add_argument("--top", type=int, default=10)
    l.add_argument("--spread", type=int, default=5)
    l.set_defaults(func=cmd_large)

    r = sub.add_parser("report", help="print the current top configs for a mode")
    r.add_argument("--mode", choices=MODES, required=True)
    r.add_argument("--top", type=int, default=10)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
