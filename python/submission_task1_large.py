"""
submission_task1_large.py — Run and plot Task 1 Large submission configurations.

Runs the 15 submission candidate configurations for the Task 1 Large dataset,
generates a Recall vs Search Time plot, and saves JSON and Markdown summaries to results/submission/task1_large.
"""
from __future__ import annotations

import json
import re
import sys
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
    non_zeros: int
    k_graph: int
    k_ext: int
    prune_worst: int
    max_dist: int
    evp_k: int
    eps_ext: float = 0.002


MODES: list[ModeConfig] = [
    # Mode 4 configurations (Tuned Large)
    ModeConfig(
        name="mode4_slot1",
        mode="evp-rerank",
        label="Slot 1 (Mode 4): Recall ~0.792",
        settings="nz=608, kg=26, kext=32, prune=9, md=500, evpK=50",
        non_zeros=608, k_graph=26, k_ext=32, prune_worst=9, max_dist=500, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot2",
        mode="evp-rerank",
        label="Slot 2 (Mode 4): Recall ~0.802",
        settings="nz=608, kg=26, kext=32, prune=9, md=600, evpK=50",
        non_zeros=608, k_graph=26, k_ext=32, prune_worst=9, max_dist=600, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot3",
        mode="evp-rerank",
        label="Slot 3 (Mode 4): Recall ~0.811",
        settings="nz=608, kg=26, kext=32, prune=9, md=700, evpK=50",
        non_zeros=608, k_graph=26, k_ext=32, prune_worst=9, max_dist=700, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot4",
        mode="evp-rerank",
        label="Slot 4 (Mode 4): Recall ~0.818",
        settings="nz=608, kg=26, kext=32, prune=9, md=800, evpK=50",
        non_zeros=608, k_graph=26, k_ext=32, prune_worst=9, max_dist=800, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot5",
        mode="evp-rerank",
        label="Slot 5 (Mode 4): Recall ~0.824",
        settings="nz=608, kg=26, kext=32, prune=9, md=900, evpK=50",
        non_zeros=608, k_graph=26, k_ext=32, prune_worst=9, max_dist=900, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot6",
        mode="evp-rerank",
        label="Slot 6 (Mode 4): Recall ~0.829",
        settings="nz=608, kg=26, kext=32, prune=9, md=1000, evpK=50",
        non_zeros=608, k_graph=26, k_ext=32, prune_worst=9, max_dist=1000, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot7",
        mode="evp-rerank",
        label="Slot 7 (Mode 4): Recall ~0.835",
        settings="nz=608, kg=26, kext=32, prune=9, md=1200, evpK=50",
        non_zeros=608, k_graph=26, k_ext=32, prune_worst=9, max_dist=1200, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot8",
        mode="evp-rerank",
        label="Slot 8 (Mode 4): Recall ~0.839",
        settings="nz=608, kg=26, kext=32, prune=9, md=1400, evpK=50",
        non_zeros=608, k_graph=26, k_ext=32, prune_worst=9, max_dist=1400, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot9",
        mode="evp-rerank",
        label="Slot 9 (Mode 4): Recall ~0.842",
        settings="nz=512, kg=32, kext=24, prune=11, md=900, evpK=50",
        non_zeros=512, k_graph=32, k_ext=24, prune_worst=11, max_dist=900, evp_k=50
    ),
    ModeConfig(
        name="mode4_slot10",
        mode="evp-rerank",
        label="Slot 10 (Mode 4): Recall ~0.847",
        settings="nz=512, kg=32, kext=24, prune=11, md=800, evpK=100",
        non_zeros=512, k_graph=32, k_ext=24, prune_worst=11, max_dist=800, evp_k=100
    ),
    
    # Mode 7 configurations (Tuned Large)
    ModeConfig(
        name="mode7_slot11",
        mode="evp-asymmetric-rerank",
        label="Slot 11 (Mode 7): Recall ~0.806",
        settings="nz=576, kg=28, kext=34, prune=10, md=400, evpK=50",
        non_zeros=576, k_graph=28, k_ext=34, prune_worst=10, max_dist=400, evp_k=50
    ),
    ModeConfig(
        name="mode7_slot12",
        mode="evp-asymmetric-rerank",
        label="Slot 12 (Mode 7): Recall ~0.815",
        settings="nz=512, kg=32, kext=24, prune=11, md=400, evpK=50",
        non_zeros=512, k_graph=32, k_ext=24, prune_worst=11, max_dist=400, evp_k=50
    ),
    ModeConfig(
        name="mode7_slot13",
        mode="evp-asymmetric-rerank",
        label="Slot 13 (Mode 7): Recall ~0.827",
        settings="nz=512, kg=32, kext=24, prune=11, md=500, evpK=50",
        non_zeros=512, k_graph=32, k_ext=24, prune_worst=11, max_dist=500, evp_k=50
    ),
    ModeConfig(
        name="mode7_slot14",
        mode="evp-asymmetric-rerank",
        label="Slot 14 (Mode 7): Recall ~0.837",
        settings="nz=512, kg=32, kext=24, prune=11, md=600, evpK=50",
        non_zeros=512, k_graph=32, k_ext=24, prune_worst=11, max_dist=600, evp_k=50
    ),
    ModeConfig(
        name="mode7_slot15",
        mode="evp-asymmetric-rerank",
        label="Slot 15 (Mode 7): Recall ~0.849",
        settings="nz=576, kg=28, kext=34, prune=10, md=800, evpK=75",
        non_zeros=576, k_graph=28, k_ext=34, prune_worst=10, max_dist=800, evp_k=75
    ),
]


def _fmt(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.2f}s"


def _fmt_recall(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val * 100.0:.2f}%"


def run_mode(runner: Task1Runner, cfg: ModeConfig, num_threads: int) -> Task1Result | None:
    print(f"\n{'='*60}")
    print(f"  Running: {cfg.label}")
    print(f"  Settings: {cfg.settings}")
    print(f"{'='*60}\n")

    kwargs: dict = dict(
        mode=cfg.mode,
        size="large",
        non_zeros=cfg.non_zeros,
        k_graph=cfg.k_graph,
        k_ext=cfg.k_ext,
        prune_worst=cfg.prune_worst,
        max_dist=cfg.max_dist,
        evp_k=cfg.evp_k,
        eps_ext=cfg.eps_ext,
        threads=num_threads,
    )

    try:
        result = runner.run(**kwargs)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None

    if not result.succeeded:
        print(f"  ERROR: container exited with code {result.exit_code}", file=sys.stderr)
        return None

    return result


def generate_outputs(results: dict[str, Task1Result], output_dir: Path, system_info: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save results to JSON
    json_data: dict = {}
    if system_info:
        json_data["system_info"] = system_info
    for name, res in results.items():
        if res is not None:
            json_data[name] = res.to_dict()
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"[submission_task1_large] Saved detailed JSON to {output_dir / 'results.json'}")

    # 2. Build plot
    plt.figure(figsize=(10, 6))
    
    color_map = {
        "mode4": "tab:green",
        "mode7": "tab:purple"
    }
    
    has_points = False
    for i, cfg in enumerate(MODES):
        res = results.get(cfg.name)
        if res is None:
            continue
            
        recall_val = res.best_recall if res.best_recall is not None else res.last_recall
        if recall_val is None:
            continue
            
        # Task 1: total time is what matters (everything runs sequentially on-the-fly)
        total_time_s = res.overall_time_s
        if total_time_s is None:
            continue
        
        mode_prefix = "mode4" if "mode4" in cfg.name else "mode7"
        color = color_map.get(mode_prefix, "tab:gray")
        
        plt.scatter(
            total_time_s, recall_val * 100.0,
            color=color, edgecolor="black", s=150, zorder=5, label=cfg.label
        )
        
        # Annotate slot number
        plt.annotate(
            str(i + 1),
            (total_time_s, recall_val * 100.0),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            weight='bold'
        )
        has_points = True

    if has_points:
        import matplotlib.ticker as ticker
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=4))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        
        plt.axhline(y=80.0, color="gray", linestyle="--", label="Target Recall (80%)")
        plt.xlabel("Total Time (s)")
        plt.ylabel("Recall @ 15 (%)")
        plt.title("Task 1 Large Submission — Recall vs Total Time")
        plt.grid(True, which="both", linestyle=":", alpha=0.6)
        
        plot_path = output_dir / "recall_vs_time.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[submission_task1_large] Saved plot to {plot_path}")

    # 3. Save Markdown Summary
    md_content = []
    md_content.append("# Task 1 Large Submission Summary — Large Dataset (6.4M vectors)")
    md_content.append("\nThis table lists the metrics for the 15 submission slots.\n")
    if system_info:
        cpu = system_info.get('cpu_model') or 'unknown'
        phys = system_info.get('cpu_physical')
        logi = system_info.get('cpu_logical')
        ram  = system_info.get('ram_total_gb')
        c_cpu = system_info.get('container_cpus')
        c_ram = system_info.get('container_ram_gb')
        cores_str = f"{phys}C/{logi}T" if phys and logi else (f"{logi}T" if logi else "?")
        md_content.append(f"**Host:** {cpu} ({cores_str}) &nbsp;·&nbsp; RAM: {ram} GiB total")
        md_content.append(f"**Container limits:** {c_cpu} CPU threads · {c_ram} GiB RAM\n")
    
    headers = ["Slot", "Mode", "Method", "Settings", "Load", "Quant", "Build", "Convert", "Explore", "Rerank", "Total", "Recall"]
    md_content.append("| " + " | ".join(headers) + " |")
    md_content.append("|" + "|".join([":---:" if i == 0 or i > 2 else "---" for i in range(len(headers))]) + "|")
    
    for i, cfg in enumerate(MODES):
        res = results.get(cfg.name)
        if res is None:
            md_content.append(f"| {i+1} | {cfg.mode} | {cfg.label} | {cfg.settings} | ERR | — | — | — | — | — | — | — |")
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
        
        row = [str(i+1), "mode4" if "rerank" in cfg.mode and "asymmetric" not in cfg.mode else "mode7", cfg.label, cfg.settings, load, quant, build, convert, explore, rerank, overall, recall]
        md_content.append("| " + " | ".join(row) + " |")

    md_output = "\n".join(md_content)
    with open(output_dir / "results.md", "w") as f:
        f.write(md_output)
    print(f"[submission_task1_large] Saved markdown summary to {output_dir / 'results.md'}")


def main() -> None:
    runner = Task1Runner(results_dir=Path(__file__).parent / "results", echo_logs=True)
    runner.build_image(force=False)
    system_info = runner.get_system_info()

    num_threads = runner.cpu_limit
    results: dict[str, Task1Result] = {}
    for cfg in MODES:
        sys.stdout.flush()
        result = run_mode(runner, cfg, num_threads)
        results[cfg.name] = result
        sys.stdout.flush()

    output_dir = Path(__file__).parent / "results" / "submission" / "task1_large"
    generate_outputs(results, output_dir, system_info)

    print("Done.")


if __name__ == "__main__":
    main()
