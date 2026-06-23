"""
evaluate_combined_task1.py

Combines the plots from Task 1 Small Benchmark and Task 1 Large Submission
side-by-side in a single figure. The first subplot takes up 2/3 of the width,
and the second subplot (without a legend) takes up 1/3 of the width.
Saves the output to the results folder as png and pdf.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Config details for the 15 slots of Task 1 Large
SLOTS_LARGE = [
    ("mode4_slot1", 1, "evp-rerank"),
    ("mode4_slot2", 2, "evp-rerank"),
    ("mode4_slot3", 3, "evp-rerank"),
    ("mode4_slot4", 4, "evp-rerank"),
    ("mode4_slot5", 5, "evp-rerank"),
    ("mode4_slot6", 6, "evp-rerank"),
    ("mode4_slot7", 7, "evp-rerank"),
    ("mode4_slot8", 8, "evp-rerank"),
    ("mode4_slot9", 9, "evp-rerank"),
    ("mode4_slot10", 10, "evp-rerank"),
    ("mode7_slot11", 11, "evp-asymmetric-rerank"),
    ("mode7_slot12", 12, "evp-asymmetric-rerank"),
    ("mode7_slot13", 13, "evp-asymmetric-rerank"),
    ("mode7_slot14", 14, "evp-asymmetric-rerank"),
    ("mode7_slot15", 15, "evp-asymmetric-rerank"),
]

def main() -> None:
    base_dir = Path(__file__).parent
    
    # Paths for Task 1 Small
    small_results_dir = base_dir / "results" / "benchmark" / "task1_small"
    small_json_path = small_results_dir / "results.json"
    
    # Paths for Task 1 Large
    large_results_dir = base_dir / "results" / "submission" / "task1_large"
    large_json_path = large_results_dir / "results.json"
    
    # Output path
    output_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path_png = output_dir / "combined_recall_vs_time.png"
    plot_path_pdf = output_dir / "combined_recall_vs_time.pdf"

    # Read data
    if not small_json_path.exists():
        print(f"Error: {small_json_path} does not exist.")
        return
    if not large_json_path.exists():
        print(f"Error: {large_json_path} does not exist.")
        return

    with open(small_json_path, "r") as f:
        small_data = json.load(f)
    with open(large_json_path, "r") as f:
        large_data = json.load(f)

    # 1. Process Task 1 Small Data
    small_approaches = {
        "Lin EVP": ("mode2", "0.5"),
        "Lin EVP Asymm": ("mode8", "0.5"),
        "DEG FP16 Baseline": ("mode1", "0"),
        "DEG EVP Baseline": ("mode3", "0.5"),
        "DEG EVP-Asym": ("mode6", "0.5"),
        "DEG EVP + Reranking": ("mode4", "0.5"),
        "DEG EVP -> FP16 repl": ("mode5", "0.5"),
    }

    small_table_data = {}
    for name, (mode_key, quant) in small_approaches.items():
        mode_data = small_data.get(mode_key)
        if not mode_data:
            continue
        small_table_data[name] = {
            "overall_time": mode_data.get("overall_time_s", 0.0),
            "best_recall": mode_data.get("best_recall", 0.0),
        }

    # 2. Process Task 1 Large Data
    large_plot_points = []
    for key, slot_num, mode_name in SLOTS_LARGE:
        res_data = large_data.get(key)
        if not res_data:
            continue
        large_plot_points.append({
            "slot_num": slot_num,
            "mode": mode_name,
            "overall_time": res_data.get("overall_time_s", 0.0),
            "recall": res_data.get("best_recall", 0.0) * 100.0
        })

    # Create figure with width ratio 2:1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150, gridspec_kw={'width_ratios': [2, 1]})

    # --- Plot Subplot 1: Task 1 Small ---
    # Draw horizontal baselines
    evp_linear_recall = small_table_data.get("Lin EVP", {}).get("best_recall", 0.7124) * 100.0
    evp_asym_linear_recall = small_table_data.get("Lin EVP Asymm", {}).get("best_recall", 0.7854) * 100.0

    ax1.axhline(y=80.0, color="gray", linestyle=":", linewidth=2.0, label="Target Recall (80%)")
    ax1.axhline(y=evp_asym_linear_recall, color="tab:purple", linestyle="-.", linewidth=1.8, 
                label=f"EVP Asym Linear Search ({evp_asym_linear_recall:.2f}%)")
    ax1.axhline(y=evp_linear_recall, color="tab:orange", linestyle="--", linewidth=1.8, 
                label=f"EVP Linear Search ({evp_linear_recall:.2f}%)")

    small_colors = {
        "DEG FP16 Baseline": "tab:blue",
        "DEG EVP Baseline": "tab:orange",
        "DEG EVP-Asym": "tab:purple",
        "DEG EVP + Reranking": "tab:green",
        "DEG EVP -> FP16 repl": "tab:red",
    }
    small_markers = {
        "DEG FP16 Baseline": "o",
        "DEG EVP Baseline": "s",
        "DEG EVP-Asym": "^",
        "DEG EVP + Reranking": "D",
        "DEG EVP -> FP16 repl": "p",
    }
    small_sizes = {
        "DEG FP16 Baseline": 180,
        "DEG EVP Baseline": 130,
        "DEG EVP-Asym": 200,
        "DEG EVP + Reranking": 130,
        "DEG EVP -> FP16 repl": 230,
    }

    for name, color in small_colors.items():
        metrics = small_table_data.get(name)
        if not metrics:
            continue
        x = metrics["overall_time"]
        y = metrics["best_recall"] * 100.0
        marker = small_markers.get(name, "o")
        size = small_sizes.get(name, 150)
        ax1.scatter(
            x, y,
            color=color, edgecolor="black", s=size, zorder=5, label=name, marker=marker
        )

    ax1.set_xlabel("Total Time (s)", fontsize=18)
    ax1.set_ylabel("Recall @ 15 (%)", fontsize=18)
    ax1.set_title("Task 1 Small Benchmark", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.grid(True, which="both", linestyle=":", alpha=0.8)
    ax1.set_ylim(65.0, 86.0)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=4))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax1.legend(loc="lower right", labelspacing=1.0, fontsize=15)

    # --- Plot Subplot 2: Task 1 Large ---
    ax2.axhline(y=80.0, color="gray", linestyle="--")

    # Filter and sort points by overall_time to draw a clean connected line
    filtered_points = [pt for pt in large_plot_points if "asymmetric" not in pt["mode"]]
    filtered_points.sort(key=lambda pt: pt["overall_time"])

    if filtered_points:
        x_coords = [pt["overall_time"] for pt in filtered_points]
        y_coords = [pt["recall"] for pt in filtered_points]
        ax2.plot(x_coords, y_coords, color="tab:green", linestyle="-", linewidth=1.5, zorder=4)

    for pt in filtered_points:
        ax2.scatter(
            pt["overall_time"], pt["recall"],
            color="tab:green", edgecolor="black", s=130, zorder=5, marker="D"
        )

    ax2.set_xlabel("Total Time (s)", fontsize=18)
    ax2.set_title("Task 1 Large Submission", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.grid(True, which="both", linestyle=":", alpha=0.8)
    ax2.set_ylim(78.0, 86.0)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=4))
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    # No legend on ax2 as requested!

    plt.tight_layout()
    plt.savefig(plot_path_png, dpi=150)
    plt.savefig(plot_path_pdf)
    plt.close()
    print(f"Successfully saved combined plots to: {plot_path_png} and {plot_path_pdf}")

if __name__ == "__main__":
    main()
