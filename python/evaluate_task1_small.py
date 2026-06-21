"""
evaluate_task1_small.py

Reads benchmark results from results.json, generates a formatted markdown table
comparing the five DEG approaches and two linear search baselines, and creates
a premium recall vs overall time plot with target and baseline lines.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def format_time_component(val: float | None) -> str:
    if val is None or val == 0.0:
        return "0"
    return f"{val:.2f}"

def format_bau_convert(build: float | None, convert: float | None) -> str:
    build_str = format_time_component(build)
    convert_str = format_time_component(convert)
    
    if convert_str != "0":
        return f"{build_str} + {convert_str}"
    return build_str

def format_expl_rerank(explore: float | None, rerank: float | None) -> str:
    explore_str = format_time_component(explore)
    rerank_str = format_time_component(rerank)
    
    if rerank_str != "0":
        return f"{explore_str} + {rerank_str}"
    return explore_str

def main() -> None:
    # Setup paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results" / "benchmark" / "task1_small"
    json_path = results_dir / "results.json"
    table_path = results_dir / "table.md"
    plot_path = results_dir / "evaluation_recall_vs_time.png"

    print(f"Reading benchmark results from: {json_path}")
    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Define approaches and their modes in JSON
    # Approach -> (Mode Key, Quantization value)
    approaches = {
        "Lin EVP": ("mode2", "0.5"),
        "Lin EVP Asymm": ("mode8", "0.5"),
        "DEG FP16 Baseline": ("mode1", "0"),
        "DEG EVP Baseline": ("mode3", "0.5"),
        "DEG EVP-Asym": ("mode6", "0.5"),
        "DEG EVP + Reranking": ("mode4", "0.5"),
        "DEG EVP -> FP16 repl": ("mode5", "0.5"),
    }

    # Generate Markdown Table
    table_lines = [
        "### Übersicht Zeiten / Recall",
        "",
        "| Ansatz | Quant | Bau+Convert | Expl + Rerank | Overall | Recall |",
        "|:---|:---:|:---:|:---:|:---:|:---:|"
    ]

    table_data = {}
    for name, (mode_key, quant) in approaches.items():
        mode_data = data.get(mode_key)
        if not mode_data:
            print(f"Warning: Mode {mode_key} not found in results.json")
            table_lines.append(f"| {name} | {quant} | — | — | — | — |")
            continue

        quant_time = mode_data.get("quant_time_s", 0.0)
        build_time = mode_data.get("build_time_s", 0.0)
        convert_time = mode_data.get("convert_time_s", 0.0)
        explore_time = mode_data.get("explore_time_s", 0.0)
        rerank_time = mode_data.get("rerank_time_s", 0.0)
        overall_time = mode_data.get("overall_time_s", 0.0)
        best_recall = mode_data.get("best_recall", 0.0)

        # Format columns
        quant_str = "0" if quant_time == 0.0 else f"{quant_time:.2f}"
        bau_convert_str = format_bau_convert(build_time, convert_time)
        expl_rerank_str = format_expl_rerank(explore_time, rerank_time)
        overall_str = f"{overall_time:.2f}"
        recall_str = f"{best_recall:.4f}"

        row_str = f"| {name} | {quant_str} | {bau_convert_str} | {expl_rerank_str} | {overall_str} | {recall_str} |"
        table_lines.append(row_str)

        # Save for plotting
        table_data[name] = {
            "overall_time": overall_time,
            "best_recall": best_recall,
            "explore_time": explore_time,
            "rerank_time": rerank_time,
        }

    # Write Markdown table
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"Successfully wrote summary table to: {table_path}")

    # Generate Plot matching original style
    plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.gca()

    # Revert to standard colors matching original benchmark script
    colors = {
        "DEG FP16 Baseline": "tab:blue",
        "DEG EVP Baseline": "tab:orange",
        "DEG EVP + Reranking": "tab:green",
        "DEG EVP -> FP16 repl": "tab:red",
        "DEG EVP-Asym": "tab:purple",
    }

    markers = {
        "DEG FP16 Baseline": "o",           # Circle
        "DEG EVP Baseline": "s",            # Square
        "DEG EVP-Asym": "^",                # Triangle Up
        "DEG EVP + Reranking": "D",           # Diamond
        "DEG EVP -> FP16 repl": "p",          # Pentagon
    }

    # Custom sizes to visually balance different geometric shapes
    sizes = {
        "DEG FP16 Baseline": 180,           # Circle
        "DEG EVP Baseline": 130,            # Square
        "DEG EVP-Asym": 200,                # Triangle Up
        "DEG EVP + Reranking": 130,           # Diamond
        "DEG EVP -> FP16 repl": 230,          # Pentagon
    }

    # Extract horizontal lines baseline recall values
    evp_linear_recall = table_data.get("Lin EVP", {}).get("best_recall", 0.7124) * 100.0
    evp_asym_linear_recall = table_data.get("Lin EVP Asymm", {}).get("best_recall", 0.7854) * 100.0

    # Draw horizontal baselines with styles and matched colors
    plt.axhline(y=80.0, color="gray", linestyle=":", linewidth=2.0, label="Target Recall (80%)")
    plt.axhline(y=evp_asym_linear_recall, color="tab:purple", linestyle="-.", linewidth=1.8, 
                label=f"EVP Asym Linear Search ({evp_asym_linear_recall:.2f}%)")
    plt.axhline(y=evp_linear_recall, color="tab:orange", linestyle="--", linewidth=1.8, 
                label=f"EVP Linear Search ({evp_linear_recall:.2f}%)")

    # Plot DEG approaches as scatter points
    for name, color in colors.items():
        metrics = table_data.get(name)
        if not metrics:
            continue
        
        x = metrics["overall_time"]
        y = metrics["best_recall"] * 100.0
        marker = markers.get(name, "o")
        size = sizes.get(name, 150)

        plt.scatter(
            x, y,
            color=color, edgecolor="black", s=size, zorder=5, label=name, marker=marker
        )

    # Style axes and grid
    plt.xlabel("Total Time (s)")
    plt.ylabel("Recall @ 15 (%)")
    plt.title("Task 1 Small Benchmark — Recall vs Total Time")
    
    # Grid configuration matching original dotted style but more visible
    plt.grid(True, which="both", linestyle=":", alpha=0.8)

    # Set Y-axis limits
    plt.ylim(65.0, 86.0)

    # Customize ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # Add legend with generous spacing
    plt.legend(loc="lower right", labelspacing=1.0)

    # Save high-res plot
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Successfully saved plot to: {plot_path}")

if __name__ == "__main__":
    main()
