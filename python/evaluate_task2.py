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
    "mode10_no_flas",
    "mode10_flas",
    "mode10_ip_flas",
    "mode5_flas",
    "mode6_flas",
]

# Mode details mapping
MODE_DETAILS = {
    "mode1_no_flas": ("mode1", "Mode 1: FP32 Build & FP32 Explore (no FLAS)"),
    "mode3_no_flas": ("mode3", "Mode 3: FP32 IP Build & FP16 IP Explore (no FLAS)"),
    "mode3_flas": ("mode3", "Mode 3: FP32 IP Build & FP16 IP Explore (+ L2 FLAS)"),
    "mode5_no_flas": ("mode5", "Mode 5: FP32 L2 Build (d+1) & FP16 IP Explore (no FLAS)"),
    "mode5_flas": ("mode5", "Mode 5: FP32 L2 Build (d+1) & FP16 IP Explore (+ L2 FLAS)"),
    "mode6_flas": ("mode6", "Mode 6: FP32 L2 Build (d+1) & FP16 L2 Explore (+ L2 FLAS)"),
    "mode10_no_flas": ("mode10", "Mode 10: FP32 IP Build (d+1) & FP16 IP Explore (no FLAS)"),
    "mode10_flas": ("mode10", "Mode 10: FP32 IP Build (d+1) & FP16 IP Explore (+ L2 FLAS)"),
    "mode10_ip_flas": ("mode10", "Mode 10: FP32 IP Build (d+1) & FP16 IP Explore (+ IP FLAS)"),
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
        return "^", "-.", 8, 160   # Triangle, dash-dot
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
    plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.gca()

    # Draw Target Recall Baseline (dotted gray line)
    plt.axhline(y=80.0, color="gray", linestyle=":", linewidth=2.0, label="Target Recall (80%)")

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

        # Plot curve
        plt.plot(
            times, recalls,
            marker=marker, label=label, linewidth=2, color=color, linestyle=linestyle, markersize=l_size
        )

        # Highlight best sweep point
        best = find_best_point(res_data["sweep_points"])
        if best:
            plt.scatter(
                [best["search_time_ms"]], [best["recall"] * 100.0],
                color=color, edgecolor="black", marker=marker, s=s_size, zorder=5
            )

    # Style axes and grid
    plt.xlabel("Search Time (ms)")
    plt.ylabel("Recall @ 30 (%)")
    plt.title("Task 2 Benchmark — Recall @ 30 vs Search Time")
    
    # Grid configuration matching original benchmark style
    plt.grid(True, which="both", linestyle=":", alpha=0.8)

    # Use log scale on x-axis to expand the fast 25ms - 40ms region
    plt.xscale('log')
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    # Explicit tick marks for clarity in the log scale
    x_ticks = [25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110]
    ax.set_xticks(x_ticks)
    ax.get_xaxis().set_minor_locator(ticker.NullLocator())

    # Add legend with spacing
    plt.legend(loc="lower right", labelspacing=1.0)

    # Save plot
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Successfully saved plot to: {plot_path}")

if __name__ == "__main__":
    main()
