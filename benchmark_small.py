"""
benchmark_small.py — Reproduce the small-dataset benchmark table from README.

Runs all 7 modes sequentially on the small dataset (200K vectors) and prints
a Markdown table identical to the one in README.md so results can be compared.

Usage
-----
    uv run python benchmark_small.py

Prerequisites
-------------
- Docker image must be built:  docker build -t sisap26-deglib .
- Docker Desktop must be running.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from docker_runner import Task1Result, Task1Runner


@dataclass
class ModeConfig:
    name: str
    mode: str
    label: str
    settings: str
    max_dist: int = 200
    evp_k: int | None = None


MODES: list[ModeConfig] = [
    ModeConfig(name="mode1", mode="fp16", label="FP16 Build+Explore", settings="M=32, MaxDist=100", max_dist=100),
    ModeConfig(name="mode2", mode="evp-linear", label="EVP linear search", settings="—"),
    ModeConfig(name="mode3", mode="evp", label="EVP Build+Explore", settings="M=32, MaxDist=200"),
    ModeConfig(name="mode4", mode="evp-rerank", label="EVP Build+Explore+Rerank", settings="M=32, MaxDist=200, evpK=50", evp_k=50),
    ModeConfig(name="mode5", mode="evp-build-fp16-external-search", label="EVP build+FP16 Explore", settings="M=32, MaxDist=200"),
    ModeConfig(name="mode6", mode="evp-asymmetric", label="EVP build+Asym Explore", settings="M=32, MaxDist=200"),
    ModeConfig(name="mode7", mode="evp-asymmetric-rerank", label="EVP build+Asym+Rerank", settings="M=32, MaxDist=200, evpK=50", evp_k=50),
]


def _fmt(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}s"


def _fmt_recall(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.2f}%"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def run_mode(runner: Task1Runner, cfg: ModeConfig) -> Task1Result | None:
    print(f"\n{'='*60}")
    print(f"  Mode {cfg.name[4:]} — {cfg.label}")
    print(f"  Settings: {cfg.settings}")
    print(f"{'='*60}\n")

    kwargs: dict = dict(mode=cfg.mode, max_dist=cfg.max_dist)
    if cfg.evp_k is not None:
        kwargs["evp_k"] = cfg.evp_k

    try:
        result = runner.run(**kwargs)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None

    if not result.succeeded:
        print(f"  ERROR: container exited with code {result.exit_code}", file=sys.stderr)
        return None

    return result


def print_table(results: dict[str, Task1Result]) -> None:
    print("\n" + "=" * 60)
    print("  Benchmark Results — Small Dataset (200K vectors)")
    print("=" * 60)

    header = f"| {'Mode':<4} | {'Method':<30} | {'Settings':<35} | {'Load':>6} | {'Quant':>6} | {'Build':>6} | {'Convert':>6} | {'Explore':>6} | {'Rerank':>6} | {'Total':>6} | {'Recall':>7} |"
    sep = "|" + ":---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"

    print()
    print(header)
    print(sep)

    for cfg in MODES:
        result = results.get(cfg.name)
        if result is None:
            print(f"| {cfg.name[4:]:<4} | {cfg.label:<30} | {cfg.settings:<35} | {'ERR':>6} | {'':>6} | {'':>6} | {'':>6} | {'':>6} | {'':>6} | {'':>6} | {'':>7} |")
            continue

        load = _fmt(result.load_time_s)
        quant = _fmt(result.quant_time_s)
        build = _fmt(result.build_time_s)
        convert = _fmt(result.convert_time_s)
        explore = _fmt(result.explore_time_s)
        rerank = _fmt(result.rerank_time_s)
        overall = _fmt(result.overall_time_s)

        recall_val = result.best_recall if result.best_recall is not None else result.last_recall
        recall = _fmt_recall(recall_val)

        print(f"| {cfg.name[4:]:<4} | {cfg.label:<30} | {cfg.settings:<35} | {load:>6} | {quant:>6} | {build:>6} | {convert:>6} | {explore:>6} | {rerank:>6} | {overall:>6} | {recall:>7} |")

    print()
    note_count = sum(1 for r in results.values() if r is not None)
    print(f"  {note_count}/{len(MODES)} modes completed successfully.")
    print()


def main() -> None:
    runner = Task1Runner(results_dir=Path("./results"), echo_logs=True)
    runner.build_image(force=False)

    results: dict[str, Task1Result] = {}

    for cfg in MODES:
        sys.stdout.flush()
        result = run_mode(runner, cfg)
        results[cfg.name] = result
        sys.stdout.flush()

    print_table(results)

    print("Done.")


if __name__ == "__main__":
    main()
