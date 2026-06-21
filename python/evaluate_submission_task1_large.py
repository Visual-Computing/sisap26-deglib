"""
evaluate_submission_task1_large.py

Reads benchmark results from results.json, generates a formatted markdown table
for the 15 submission slots of Task 1 Large, and creates a clean recall vs total time
plot with slot annotations and a grouped legend.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Config details for the 15 slots
SLOTS = [
    # Mode 4 configurations (Slots 1-10)
    ("mode4_slot1", 1, "evp-rerank", "Slot 1 (Mode 4): Recall ~0.792", "nz=608, kg=26, kext=32, prune=9, md=500, evpK=50"),
    ("mode4_slot2", 2, "evp-rerank", "Slot 2 (Mode 4): Recall ~0.802", "nz=608, kg=26, kext=32, prune=9, md=600, evpK=50"),
    ("mode4_slot3", 3, "evp-rerank", "Slot 3 (Mode 4): Recall ~0.811", "nz=608, kg=26, kext=32, prune=9, md=700, evpK=50"),
    ("mode4_slot4", 4, "evp-rerank", "Slot 4 (Mode 4): Recall ~0.818", "nz=608, kg=26, kext=32, prune=9, md=800, evpK=50"),
    ("mode4_slot5", 5, "evp-rerank", "Slot 5 (Mode 4): Recall ~0.824", "nz=608, kg=26, kext=32, prune=9, md=900, evpK=50"),
    ("mode4_slot6", 6, "evp-rerank", "Slot 6 (Mode 4): Recall ~0.829", "nz=608, kg=26, kext=32, prune=9, md=1000, evpK=50"),
    ("mode4_slot7", 7, "evp-rerank", "Slot 7 (Mode 4): Recall ~0.835", "nz=608, kg=26, kext=32, prune=9, md=1200, evpK=50"),
    ("mode4_slot8", 8, "evp-rerank", "Slot 8 (Mode 4): Recall ~0.839", "nz=608, kg=26, kext=32, prune=9, md=1400, evpK=50"),
    ("mode4_slot9", 9, "evp-rerank", "Slot 9 (Mode 4): Recall ~0.842", "nz=512, kg=32, kext=24, prune=11, md=900, evpK=50"),
    ("mode4_slot10", 10, "evp-rerank", "Slot 10 (Mode 4): Recall ~0.847", "nz=512, kg=32, kext=24, prune=11, md=800, evpK=100"),
    # Mode 7 configurations (Slots 11-15)
    ("mode7_slot11", 11, "evp-asymmetric-rerank", "Slot 11 (Mode 7): Recall ~0.806", "nz=576, kg=28, kext=34, prune=10, md=400, evpK=50"),
    ("mode7_slot12", 12, "evp-asymmetric-rerank", "Slot 12 (Mode 7): Recall ~0.815", "nz=512, kg=32, kext=24, prune=11, md=400, evpK=50"),
    ("mode7_slot13", 13, "evp-asymmetric-rerank", "Slot 13 (Mode 7): Recall ~0.827", "nz=512, kg=32, kext=24, prune=11, md=500, evpK=50"),
    ("mode7_slot14", 14, "evp-asymmetric-rerank", "Slot 14 (Mode 7): Recall ~0.837", "nz=512, kg=32, kext=24, prune=11, md=600, evpK=50"),
    ("mode7_slot15", 15, "evp-asymmetric-rerank", "Slot 15 (Mode 7): Recall ~0.849", "nz=576, kg=28, kext=34, prune=10, md=800, evpK=75"),
]

def format_time_s(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}s"

def format_time_s_precision(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.2f}s"

def main() -> None:
    # Setup paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results" / "submission" / "task1_large"
    json_path = results_dir / "results.json"
    table_path = results_dir / "table.md"
    plot_path = results_dir / "evaluation_recall_vs_time.png"

    print(f"Reading submission results from: {json_path}")
    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Generate Markdown Table
    table_lines = [
        "### Übersicht Zeiten / Recall (Task 1 Large Submission)",
        "",
        "| Slot | Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Total | Recall |",
        "|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    plot_points = []

    for key, slot_num, mode_name, label, settings in SLOTS:
        res_data = data.get(key)
        if not res_data:
            print(f"Warning: Configuration {key} not found in results.json")
            table_lines.append(f"| {slot_num} | {mode_name} | {label} | {settings} | ERR | — | — | — | — | — | — | — |")
            continue

        load = format_time_s_precision(res_data.get("load_time_s"))
        quant = format_time_s_precision(res_data.get("quant_time_s"))
        build = format_time_s_precision(res_data.get("build_time_s"))
        convert = format_time_s_precision(res_data.get("convert_time_s"))
        explore = format_time_s_precision(res_data.get("explore_time_s"))
        rerank = format_time_s_precision(res_data.get("rerank_time_s"))
        overall = format_time_s_precision(res_data.get("overall_time_s"))
        
        best_recall = res_data.get("best_recall", 0.0)
        recall_str = f"{best_recall * 100.0:.2f}%"

        row = [
            str(slot_num),
            "mode4" if "rerank" in mode_name and "asymmetric" not in mode_name else "mode7",
            label,
            settings,
            load,
            quant,
            build,
            convert,
            explore,
            rerank,
            overall,
            recall_str
        ]
        table_lines.append("| " + " | ".join(row) + " |")

        overall_time = res_data.get("overall_time_s", 0.0)
        plot_points.append({
            "slot_num": slot_num,
            "mode": mode_name,
            "overall_time": overall_time,
            "recall": best_recall * 100.0
        })

    # Write Markdown Table
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"Successfully wrote summary table to: {table_path}")

    # Generate Plot
    plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.gca()

    # Draw Target Recall Baseline (dotted gray line)
    plt.axhline(y=80.0, color="gray", linestyle="--", label="Target Recall (80%)")

    # Tracking whether we've added legend entries
    first_mode4 = True
    first_mode7 = True

    for pt in plot_points:
        x = pt["overall_time"]
        y = pt["recall"]
        slot_num = pt["slot_num"]

        if "asymmetric" not in pt["mode"]:
            # Mode 4 (evp-rerank) -> Green circle
            color = "tab:green"
            marker = "o"
            size = 180
            label = "Mode 4: evp-rerank" if first_mode4 else ""
            first_mode4 = False
        else:
            # Mode 7 (evp-asymmetric-rerank) -> Purple square
            color = "tab:purple"
            marker = "s"
            size = 130
            label = "Mode 7: evp-asymmetric-rerank" if first_mode7 else ""
            first_mode7 = False

        plt.scatter(
            x, y,
            color=color, edgecolor="black", s=size, zorder=5, label=label, marker=marker
        )

        # Annotate slot number above the point with a clean bounding box
        plt.annotate(
            str(slot_num),
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8.5,
            weight='bold',
            color="#1e293b",
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='#e2e8f0', lw=0.8)
        )

    # Style axes and grid
    plt.xlabel("Total Time (s)")
    plt.ylabel("Recall @ 15 (%)")
    plt.title("Task 1 Large Submission — Recall vs Total Time")
    
    # Grid configuration matching standard benchmark style
    plt.grid(True, which="both", linestyle=":", alpha=0.8)

    # Auto ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    plt.ylim(78.0, 86.0)

    # Add legend with spacing
    plt.legend(loc="lower right", labelspacing=1.0)

    # Save plot
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Successfully saved plot to: {plot_path}")

if __name__ == "__main__":
    main()
