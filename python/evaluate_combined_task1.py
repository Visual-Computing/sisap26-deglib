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
import matplotlib.transforms as mtransforms

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
    plot_path_png = output_dir / "combined_recall_vs_time_task1.png"
    plot_path_pdf = output_dir / "combined_recall_vs_time_task1.pdf"

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
        
        quant_time = mode_data.get("quant_time_s", 0.0)
        build_time = mode_data.get("build_time_s", 0.0)
        convert_time = mode_data.get("convert_time_s", 0.0)
        explore_time = mode_data.get("explore_time_s", 0.0)
        rerank_time = mode_data.get("rerank_time_s", 0.0)

        build_sum = build_time + convert_time

        small_table_data[name] = {
            "overall_time": mode_data.get("overall_time_s", 0.0),
            "best_recall": mode_data.get("best_recall", 0.0),
            "quant": "-" if quant_time == 0.0 else f"{quant_time:.2f}",
            "build": "-" if build_sum == 0.0 else f"{build_sum:.2f}",
            "explore": "-" if explore_time == 0.0 else f"{explore_time:.2f}",
            "rerank": "-" if rerank_time == 0.0 else f"{rerank_time:.2f}",
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

    # Create figure with GridSpec to include a LaTeX-style table at the top
    fig = plt.figure(figsize=(15, 11), dpi=150)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, height_ratios=[1.8, 3], width_ratios=[2, 1], figure=fig)
    ax_table = fig.add_subplot(gs[0, :])
    
    # Sub-grid for bottom-left: split ax1 into left (DEG) and right (Linear) subplots
    gs_bottom_left = gs[1, 0].subgridspec(1, 2, width_ratios=[2.5, 1], wspace=0.08)
    ax1_left = fig.add_subplot(gs_bottom_left[0, 0])
    ax1_right = fig.add_subplot(gs_bottom_left[0, 1], sharey=ax1_left)
    
    ax2 = fig.add_subplot(gs[1, 1])

    # Draw LaTeX-style table on ax_table
    ax_table.axis('off')
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)

    # Table columns layout with a visual gap between columns 4 (Reranking) and 5 (Quant)
    col_x = [-0.015, 0.26, 0.37, 0.47, 0.57, 0.69, 0.78, 0.87, 0.96]
    col_align = ['left', 'center', 'center', 'center', 'center', 'right', 'right', 'right', 'right']
    data_x = [-0.015, 0.26, 0.37, 0.47, 0.57, 0.72, 0.805, 0.895, 0.98]
    headers = [
        "Approach",
        "Graph\nConstruction",
        "Database\nVector",
        "Query\nVector",
        "Reranking\nStrategy",
        "Quant\n[s]",
        "Build\n[s]",
        "Explore\n[s]",
        "Rerank\n[s]"
    ]

    # Draw horizontal rules (toprule, midrule, bottomrule) broken into two parts
    # Part 1: General configurations (columns 1-5)
    ax_table.plot([-0.05, 0.61], [0.94, 0.94], color="black", linewidth=2.0, clip_on=False)
    ax_table.plot([-0.05, 0.61], [0.76, 0.76], color="black", linewidth=1.0, clip_on=False)
    ax_table.plot([-0.05, 0.61], [0.02, 0.02], color="black", linewidth=2.0, clip_on=False)

    # Part 2: Small Benchmark timings (columns 6-9)
    ax_table.plot([0.64, 1.00], [0.94, 0.94], color="black", linewidth=2.0, clip_on=False)
    ax_table.plot([0.64, 1.00], [0.76, 0.76], color="black", linewidth=1.0, clip_on=False)
    ax_table.plot([0.64, 1.00], [0.02, 0.02], color="black", linewidth=2.0, clip_on=False)

    # Draw header text (with centered headers for the last four columns)
    header_align = ['left', 'center', 'center', 'center', 'center', 'center', 'center', 'center', 'center']
    for x, align, text in zip(col_x, header_align, headers):
        ax_table.text(x, 0.85, text, ha=align, va='center', fontsize=15, weight='bold', clip_on=False)

    # Data rows matching mapping keys in small_table_data
    rows = [
        ("tab:orange", "X", "linear EVP", "-", "EVP", "EVP", "no", "Lin EVP"),
        ("tab:purple", "P", "linear EVP Asym", "-", "EVP", "FP16", "no", "Lin EVP Asymm"),
        ("tab:blue", "o", "DEG FP16", "FP16", "FP16", "FP16", "no", "DEG FP16 Baseline"),
        ("tab:orange", "s", "DEG EVP", "EVP", "EVP", "EVP", "no", "DEG EVP Baseline"),
        ("tab:purple", "^", "DEG EVP Asym", "EVP", "EVP", "FP16", "no", "DEG EVP-Asym"),
        ("tab:green", "D", "DEG EVP + Reranking", "EVP", "EVP", "EVP", "yes", "DEG EVP + Reranking"),
        ("tab:red", "p", "DEG EVP \u2192 FP16 repl.", "EVP", "FP16", "FP16", "no", "DEG EVP -> FP16 repl")
    ]

    y_coords = [0.66, 0.56, 0.46, 0.36, 0.26, 0.16, 0.06]
    for y, (color, style, app_name, graph, db, query, rerank, key) in zip(y_coords, rows):
        # Draw Symbol directly in the Approach column space
        ax_table.scatter(-0.04, y, color=color, marker=style, s=100, edgecolor="black", zorder=5, clip_on=False)
        # Draw Approach name closer to the symbol
        ax_table.text(-0.015, y, app_name, ha='left', va='center', fontsize=17, clip_on=False)
        
        data_entry = small_table_data.get(key, {"quant": "-", "build": "-", "explore": "-", "rerank": "-"})
        row_text = [
            graph, 
            db, 
            query, 
            rerank, 
            data_entry["quant"], 
            data_entry["build"], 
            data_entry["explore"],
            data_entry["rerank"]
        ]
        for x, align, text in zip(data_x[1:], col_align[1:], row_text):
            ax_table.text(x, y, text, ha=align, va='center', fontsize=17, clip_on=False)

    # --- Plot Subplot 1: Task 1 Small ---
    # Draw horizontal baselines and label them directly
    trans_left = mtransforms.blended_transform_factory(ax1_left.transAxes, ax1_left.transData)
    trans_right = mtransforms.blended_transform_factory(ax1_right.transAxes, ax1_right.transData)
    
    # Target Recall 80% line drawn on both split plots
    ax1_left.axhline(y=80.0, color="gray", linestyle=":", linewidth=2.0)
    ax1_right.axhline(y=80.0, color="gray", linestyle=":", linewidth=2.0)
    ax1_right.text(0.98, 80.0 + 0.3, "Target Recall (80%)", transform=trans_right, color="gray",
                   va="bottom", ha="right", fontsize=15, weight='bold')

    small_colors = {
        "DEG FP16 Baseline": "tab:blue",
        "DEG EVP Baseline": "tab:orange",
        "DEG EVP-Asym": "tab:purple",
        "DEG EVP + Reranking": "tab:green",
        "DEG EVP -> FP16 repl": "tab:red",
        "Lin EVP": "tab:orange",
        "Lin EVP Asymm": "tab:purple",
    }
    small_markers = {
        "DEG FP16 Baseline": "o",
        "DEG EVP Baseline": "s",
        "DEG EVP-Asym": "^",
        "DEG EVP + Reranking": "D",
        "DEG EVP -> FP16 repl": "p",
        "Lin EVP": "X",
        "Lin EVP Asymm": "P",
    }
    small_sizes = {
        "DEG FP16 Baseline": 180,
        "DEG EVP Baseline": 130,
        "DEG EVP-Asym": 200,
        "DEG EVP + Reranking": 130,
        "DEG EVP -> FP16 repl": 230,
        "Lin EVP": 160,
        "Lin EVP Asymm": 180,
    }

    # Plot on left or right subplot depending on the time range
    for name, color in small_colors.items():
        metrics = small_table_data.get(name)
        if not metrics:
            continue
        x = metrics["overall_time"]
        y = metrics["best_recall"] * 100.0
        marker = small_markers.get(name, "o")
        size = small_sizes.get(name, 150)
        
        target_ax = ax1_left if x < 15.0 else ax1_right
        target_ax.scatter(
            x, y,
            color=color, edgecolor="black", s=size, zorder=5, label=name, marker=marker
        )

    # Configure axes limits and scales
    ax1_left.set_xlim(4.5, 10.0)
    ax1_left.set_xticks([4.8, 5.6, 6.4, 7.2, 8.0, 8.8, 9.6])
    ax1_right.set_xscale('log')
    ax1_right.set_xlim(100, 1100)

    # Configure grid and labels
    ax1_left.set_xlabel("Total Time (s)", fontsize=21)
    ax1_left.set_ylabel("Recall @ 15 [%]", fontsize=21)
    ax1_left.set_title("Task 1 Small Benchmark", fontsize=23)
    ax1_left.tick_params(axis='both', which='major', labelsize=17)
    ax1_left.grid(True, which="both", linestyle=":", alpha=0.8)
    ax1_left.set_ylim(65.0, 86.0)
    ax1_left.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    
    ax1_right.tick_params(axis='x', which='major', labelsize=17)
    ax1_right.tick_params(axis='y', left=False, labelleft=False)
    ax1_right.grid(True, which="both", linestyle=":", alpha=0.8)

    # Hide spines between split plots
    ax1_left.spines['right'].set_visible(False)
    ax1_right.spines['left'].set_visible(False)

    # Draw broken axis diagonal marks on the spines
    d = .015  # size of the break mark
    kwargs = dict(transform=ax1_left.transAxes, color='black', clip_on=False, linewidth=1.5)
    ax1_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    # Scale width offset for the right plot as it is narrower (ratio 2.5:1)
    kwargs.update(transform=ax1_right.transAxes)
    ax1_right.plot((-d * 2.5, +d * 2.5), (-d, +d), **kwargs)
    ax1_right.plot((-d * 2.5, +d * 2.5), (1 - d, 1 + d), **kwargs)

    # Draw a prominent vertical break line and slashes in the center of the gap
    gap_x = 1.028  # center of the gap in ax1_left.transAxes coordinates
    ax1_left.plot([gap_x, gap_x], [0, 1], color='gray', linestyle='--', linewidth=1.5, transform=ax1_left.transAxes, clip_on=False)
    for y_center in [0.25, 0.50, 0.75]:
        ax1_left.plot([gap_x - 0.015, gap_x + 0.015], [y_center - 0.02, y_center + 0.02], color='black', linewidth=1.5, transform=ax1_left.transAxes, clip_on=False)
        ax1_left.plot([gap_x - 0.015, gap_x + 0.015], [y_center - 0.01, y_center + 0.03], color='black', linewidth=1.5, transform=ax1_left.transAxes, clip_on=False)

    # --- Plot Subplot 2: Task 1 Large ---
    trans2 = mtransforms.blended_transform_factory(ax2.transAxes, ax2.transData)
    ax2.axhline(y=80.0, color="gray", linestyle="--")
    ax2.text(0.98, 80.0 + 0.15, "Target Recall (80%)", transform=trans2, color="gray",
             va="bottom", ha="right", fontsize=15, weight='bold')

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

    ax2.set_xlabel("Total Time (s)", fontsize=21)
    ax2.set_title("Task 1 Large Submission", fontsize=23)
    ax2.tick_params(axis='both', which='major', labelsize=17)
    ax2.grid(True, which="both", linestyle=":", alpha=0.8)
    ax2.set_ylim(78.0, 86.0)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=4))
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    # No legend on ax2 as requested!

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.28)
    plt.savefig(plot_path_png, dpi=150)
    plt.savefig(plot_path_pdf)
    plt.close()
    print(f"Successfully saved combined plots to: {plot_path_png} and {plot_path_pdf}")

if __name__ == "__main__":
    main()
