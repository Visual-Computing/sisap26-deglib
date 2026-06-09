"""
benchmark_task1_large.py — Reproduce the large-dataset benchmark table from README.

Runs all 6 modes sequentially on the large dataset (6.4M vectors) and prints
a Markdown table identical to the one in README.md so results can be compared.

Usage
-----
    uv run python benchmark_task1_large.py

Prerequisites
-------------
- Docker Desktop must be running.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

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

    kwargs: dict = dict(mode=cfg.mode, size="large", max_dist=cfg.max_dist)
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
    print("  Benchmark Results — Large Dataset (6.4M vectors)")
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


def generate_outputs(results: dict[str, Task1Result], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save results to JSON
    json_data = {}
    for name, res in results.items():
        if res is not None:
            json_data[name] = res.to_dict()
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"[benchmark_task1_large] Saved detailed JSON to {output_dir / 'results.json'}")

    # 2. Build plot
    plt.figure(figsize=(10, 6))
    
    color_map = {
        "mode1": "tab:blue",
        "mode2": "tab:gray",
        "mode3": "tab:orange",
        "mode4": "tab:green",
        "mode5": "tab:red",
        "mode6": "tab:purple",
        "mode7": "tab:brown"
    }
    
    has_points = False
    for cfg in MODES:
        res = results.get(cfg.name)
        if res is None:
            continue
            
        recall_val = res.best_recall if res.best_recall is not None else res.last_recall
        if recall_val is None:
            continue
            
        # Search Time = explore_time + rerank_time (in seconds)
        search_time_s = (res.explore_time_s or 0.0) + (res.rerank_time_s or 0.0)
        
        color = color_map.get(cfg.name, "tab:gray")
        plt.scatter(
            search_time_s, recall_val * 100.0,
            color=color, edgecolor="black", s=150, zorder=5, label=cfg.label
        )
        
        # Annotate point
        plt.annotate(
            cfg.name.upper(),
            (search_time_s, recall_val * 100.0),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            weight='bold'
        )
        has_points = True

    if has_points:
        plt.axhline(y=80.0, color="gray", linestyle="--", label="Target Recall (80%)")
        plt.xlabel("Search Time (s)")
        plt.ylabel("Recall @ 15 (%)")
        plt.title("Task 1 Large Benchmark — Recall vs Search Time")
        plt.legend(loc="lower right")
        plt.grid(True, which="both", linestyle=":", alpha=0.6)
        
        plot_path = output_dir / "recall_vs_time.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[benchmark_task1_large] Saved plot to {plot_path}")

    # 3. Print & Save Markdown Summary
    md_content = []
    md_content.append("# Task 1 Large Benchmark Summary — Large Dataset (6.4M vectors)")
    md_content.append("\nThis table lists the metrics for each benchmark mode.\n")
    
    headers = ["Mode", "Method", "Settings", "Load", "Quant", "Build", "Convert", "Explore", "Rerank", "Total", "Recall"]
    md_content.append("| " + " | ".join(headers) + " |")
    md_content.append("|" + "|".join([":---:" if i == 0 or i > 2 else "---" for i in range(len(headers))]) + "|")
    
    for cfg in MODES:
        res = results.get(cfg.name)
        if res is None:
            md_content.append(f"| {cfg.name[4:]} | {cfg.label} | {cfg.settings} | ERR | — | — | — | — | — | — | — |")
            continue
            
        load = _fmt(res.load_time_s)
        quant = _fmt(res.quant_time_s)
        build = _fmt(res.build_time_s)
        convert = _fmt(res.convert_time_s)
        explore = _fmt(res.explore_time_s)
        rerank = _fmt(res.rerank_time_s)
        overall = _fmt(res.overall_time_s)
        
        recall_val = res.best_recall if res.best_recall is not None else res.last_recall
        recall = _fmt_recall(recall_val)
        
        row = [cfg.name[4:], cfg.label, cfg.settings, load, quant, build, convert, explore, rerank, overall, recall]
        md_content.append("| " + " | ".join(row) + " |")

    md_output = "\n".join(md_content)
    with open(output_dir / "results.md", "w") as f:
        f.write(md_output)
    print(f"[benchmark_task1_large] Saved markdown summary to {output_dir / 'results.md'}")


def main() -> None:
    runner = Task1Runner(results_dir=Path(__file__).parent / "results", echo_logs=True)
    runner.build_image(force=False)

    results: dict[str, Task1Result] = {}

    for cfg in MODES:
        sys.stdout.flush()
        result = run_mode(runner, cfg)
        results[cfg.name] = result
        sys.stdout.flush()

    print_table(results)
    
    output_dir = Path(__file__).parent / "results" / "benchmark" / "task1_large"
    generate_outputs(results, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
