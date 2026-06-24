"""
evaluate_task2.py

Reads benchmark results from results.json, filters specific modes, generates
a formatted markdown table for Task 2, and plots sweep curves with color/marker
consistency and a target recall baseline line.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Target modes to evaluate
TARGET_MODES = [
    "mode3_no_flas",
    "mode3_flas",
    "mode3_ip_flas",
    "mode10_no_flas",
    "mode10_flas",
    "mode10_ip_flas",
    "mode5_flas",
    "mode6_flas",
]

# Mode details mapping
MODE_DETAILS = {
    "mode1_no_flas": ("mode1", "Mode 1: FP32 Build & FP32 Search (no FLAS)"),
    "mode3_no_flas": ("mode3", "Mode 3: FP32 IP Build & FP16 IP Search (no FLAS)"),
    "mode3_flas": ("mode3", "Mode 3: FP32 IP Build & FP16 IP Search (+ L2 FLAS)"),
    "mode3_ip_flas": ("mode3", "Mode 3: FP32 IP Build & FP16 IP Search (+ IP FLAS)"),
    "mode5_no_flas": ("mode5", "Mode 5: FP32 L2 Build (d+1) & FP16 IP Search (no FLAS)"),
    "mode5_flas": ("mode5", "Mode 5: FP32 L2 Build (d+1) & FP16 IP Search (+ L2 FLAS)"),
    "mode6_flas": ("mode6", "Mode 6: FP32 L2 Build (d+1) & FP16 L2 Search (+ L2 FLAS)"),
    "mode10_no_flas": ("mode10", "Mode 10: FP32 IP Build (d+1) & FP16 IP Search (no FLAS)"),
    "mode10_flas": ("mode10", "Mode 10: FP32 IP Build (d+1) & FP16 IP Search (+ L2 FLAS)"),
    "mode10_ip_flas": ("mode10", "Mode 10: FP32 IP Build (d+1) & FP16 IP Search (+ IP FLAS)"),
}

# Style maps for visual consistency and color-blind friendliness
COLOR_MAP = {
    "mode1": "tab:blue",
    "mode3": "tab:orange",
    "mode5": "tab:green",
    "mode6": "tab:red",
    "mode10": "tab:purple",
}

def get_variant_style(name: str) -> tuple[str, str, int, int]:
    # Returns (marker, linestyle, line_marker_size, scatter_marker_size)
    if "no_flas" in name:
        return "o", "--", 7, 140   # Circle, dashed
    elif "ip_flas" in name:
        return "^", ":", 8, 160    # Triangle, dotted
    else:
        return "s", "-", 6, 100    # Square, solid (flas)

def format_time_s(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.2f}s"

def find_best_point(sweep_points: list[dict[str, Any]], target_recall: float = 0.8) -> dict[str, Any] | None:
    """Find the point with search_time_ms minimized among those with recall >= target_recall."""
    valid_points = [p for p in sweep_points if p.get("recall", 0.0) >= target_recall]
    if valid_points:
        return min(valid_points, key=lambda p: p.get("search_time_ms", float("inf")))
    
    # Fallback: return point with highest recall if target not reached
    if sweep_points:
        return max(sweep_points, key=lambda p: p.get("recall", 0.0))
    return None

def main() -> None:
    # Setup paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results" / "benchmark" / "task2"
    json_path = results_dir / "results.json"
    table_path = results_dir / "table.md"
    plot_path = results_dir / "evaluation_recall_vs_time.png"

    print(f"Reading benchmark results from: {json_path}")
    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Generate Markdown Table
    table_lines = [
        "### Übersicht Zeiten / Recall (Task 2)",
        "",
        "| Mode | Method | Best Settings | Load Time | Build Time | FLAS Time | Total Time | Search Time | Recall |",
        "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for name in TARGET_MODES:
        res_data = data.get(name)
        if not res_data:
            print(f"Warning: Mode {name} not found in results.json")
            continue

        mode_num, label = MODE_DETAILS[name]
        
        # Find best sweep point
        sweep_points = res_data.get("sweep_points", [])
        best = find_best_point(sweep_points)
        
        if best:
            best_settings = f"eps={best.get('eps_search', '—')}, max_dist={best.get('max_dist', '—')}"
            search_time_str = f"{best.get('search_time_ms', 0.0):.2f} ms"
            recall_str = f"{best.get('recall', 0.0) * 100.0:.2f}%"
        else:
            best_settings = "—"
            search_time_str = "—"
            recall_str = "—"

        load_str = format_time_s(res_data.get("load_time_s"))
        build_str = format_time_s(res_data.get("build_time_s"))
        flas_str = format_time_s(res_data.get("flas_time_s"))
        overall_str = format_time_s(res_data.get("overall_time_s"))

        row = [
            mode_num,
            label,
            best_settings,
            load_str,
            build_str,
            flas_str,
            overall_str,
            search_time_str,
            recall_str
        ]
        table_lines.append("| " + " | ".join(row) + " |")

    # Write Markdown Table
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"Successfully wrote summary table to: {table_path}")

    # Generate Plot matching original style but filtered
    plt.figure(figsize=(12, 8), dpi=150)
    ax = plt.gca()

    # Draw Target Recall Baseline (dotted gray line)
    plt.axhline(y=80.0, color="gray", linestyle=":", linewidth=2.5, label="Target Recall (80%)")

    # Plot curves for target modes
    for name in TARGET_MODES:
        res_data = data.get(name)
        if not res_data or not res_data.get("sweep_points"):
            continue

        mode_num, label = MODE_DETAILS[name]
        color = COLOR_MAP.get(mode_num, "tab:gray")
        marker, linestyle, l_size, s_size = get_variant_style(name)

        # Extract and sort points by search time
        pts = sorted(res_data["sweep_points"], key=lambda p: p.get("search_time_ms", 0.0))
        times = [p["search_time_ms"] for p in pts]
        recalls = [p["recall"] * 100.0 for p in pts]

        # Clean label for legend to remove "Mode X: " prefix
        clean_label = label.split(":", 1)[1].strip() if ":" in label else label

        # Plot curve
        plt.plot(
            times, recalls,
            marker=marker, label=clean_label, linewidth=2.5, color=color, linestyle=linestyle, markersize=l_size + 2
        )

        # Highlight best sweep point
        best = find_best_point(res_data["sweep_points"])
        if best:
            plt.scatter(
                [best["search_time_ms"]], [best["recall"] * 100.0],
                color=color, edgecolor="black", marker=marker, s=s_size + 40, zorder=5
            )

    # Style axes and grid
    plt.xlabel("Search Time (ms)", fontsize=20, labelpad=10)
    plt.ylabel("Recall @ 30 (%)", fontsize=20, labelpad=10)
    plt.title("Task 2 Benchmark — Recall @ 30 vs Search Time", fontsize=22, pad=15)
    
    # Grid configuration matching original benchmark style
    plt.grid(True, which="both", linestyle=":", alpha=0.8)

    # Use log scale on x-axis to expand the fast 25ms - 40ms region
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    # Explicit tick marks for clarity in the log scale
    x_ticks = [25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110]
    ax.set_xticks(x_ticks)
    ax.get_xaxis().set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Construct two legends to look like one single box
    import matplotlib.lines as mlines
    from matplotlib.offsetbox import HPacker

    # 1. Base Configs Legend (Colors)
    base_labels = {
        "mode3": "FP32 IP Build & FP16 IP Search",
        "mode10": "FP32 IP Build (d+1) & FP16 IP Search",
        "mode5": "FP32 L2 Build (d+1) & FP16 IP Search",
        "mode6": "FP32 L2 Build (d+1) & FP16 L2 Search",
    }
    base_handles = [
        mlines.Line2D([], [], color="gray", linestyle=":", linewidth=2.5, label="Target Recall (80%)")
    ]
    legend_order = ["mode3", "mode10", "mode5", "mode6"]
    for mode_num in legend_order:
        if mode_num in base_labels:
            color = COLOR_MAP.get(mode_num, "tab:gray")
            base_handles.append(
                mlines.Line2D([], [], color=color, linewidth=3, label=base_labels[mode_num])
            )

    # 2. Variant Styles Legend (temporary)
    style_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="--", linewidth=2.5, markersize=9, label="no FLAS"),
        mlines.Line2D([], [], color="gray", marker="s", linestyle="-", linewidth=2.5, markersize=8, label="+ L2 FLAS"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle=":", linewidth=2.5, markersize=10, label="+ IP FLAS"),
    ]

    # Create temporary legend with ncol=3
    leg_temp = ax.legend(handles=style_handles, ncol=3, fontsize=14, handlelength=3.5)
    main_packer = leg_temp._legend_box.get_children()[1]
    style_row = HPacker(pad=5, sep=20, children=list(main_packer.get_children()))
    leg_temp.remove()

    # Add main legend to lower right
    leg1 = ax.legend(handles=base_handles, loc="lower right", labelspacing=0.8, fontsize=15, handlelength=3.0)
    ax.add_artist(leg1)

    # Append style_row to the bottom of the main legend's column packer
    column_packer = leg1._legend_box.get_children()[1].get_children()[0]
    column_packer.get_children().append(style_row)

    # Save plot
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plot_path_pdf = plot_path.with_suffix(".pdf")
    plt.savefig(plot_path_pdf)
    plt.close()
    print(f"Successfully saved plots to: {plot_path} and {plot_path_pdf}")

if __name__ == "__main__":
    main()
