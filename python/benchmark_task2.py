"""
benchmark_task2.py — Run and plot Task 2 benchmark configurations.

This script runs the 7 benchmark configurations on the llama-dev dataset,
automatically identifies the best sweep point reaching >= 0.8 recall,
generates a Recall vs Search Time plot, and saves all outputs
(plot, JSON, and Markdown summary) to results/task2.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

from docker_runner import Task2Result, Task2Runner


@dataclass
class ModeConfig:
    name: str
    mode: str
    label: str
    settings: str
    k_ext: int = 64
    k_graph: int = 32
    eps_ext: float = 0.001
    build_threads: int = 1
    max_dist: str = ""
    eps_search: str = ""
    num_runs: int = 1
    use_flas: bool = False


MODES: list[ModeConfig] = [
    ModeConfig(
        name="mode3_no_flas",
        mode="mode3",
        label="Mode 3: FP32 Build & FP16 Explore (no FLAS)",
        settings="k_ext=64, k_graph=32, runs=3",
        max_dist="15000,20000,25000,30000",
        eps_search="0.25",
        num_runs=3,
        use_flas=False,
    ),
    ModeConfig(
        name="mode3_flas",
        mode="mode3",
        label="Mode 3: FP32 Build & FP16 Explore (+ FLAS)",
        settings="k_ext=64, k_graph=32, runs=3",
        max_dist="15000,18000,20000,23000,25000,27000,30000",
        eps_search="0.28",
        num_runs=3,
        use_flas=True,
    ),
    ModeConfig(
        name="mode5_no_flas",
        mode="mode5",
        label="Mode 5: L2 Build (d+1) & FP16 IP Explore (no FLAS)",
        settings="k_ext=64, k_graph=32, runs=10",
        max_dist="5000,6000,7000,8000,9000,10000",
        eps_search="0.18",
        num_runs=10,
        use_flas=False,
    ),
    ModeConfig(
        name="mode5_flas",
        mode="mode5",
        label="Mode 5: L2 Build (d+1) & FP16 IP Explore (+ FLAS)",
        settings="k_ext=64, k_graph=32, runs=10",
        max_dist="5000,6000,7000,8000",
        eps_search="0.18",
        num_runs=10,
        use_flas=True,
    ),
    ModeConfig(
        name="mode4_flas",
        mode="mode4",
        label="Mode 4: L2 Build (d+1) & FP32 L2 Explore (+ FLAS)",
        settings="k_ext=64, k_graph=32, runs=10",
        max_dist="5000,5500,6000,6500,7000",
        eps_search="0.008",
        num_runs=10,
        use_flas=True,
    ),
    ModeConfig(
        name="mode6_flas",
        mode="mode6",
        label="Mode 6: L2 Build (d+1) & FP16 L2 Explore (+ FLAS)",
        settings="k_ext=64, k_graph=32, runs=10",
        max_dist="5000,5500,6000,6500,7000,8000,9000,10000",
        eps_search="0.007",
        num_runs=10,
        use_flas=True,
    ),
    ModeConfig(
        name="mode7_flas",
        mode="mode7",
        label="Mode 7: L2 Build (d+2) & FP16 L2 Explore (+ FLAS)",
        settings="k_ext=64, k_graph=32, runs=10",
        max_dist="5000,5500,6000,6200,6300,6500,7000",
        eps_search="0.007",
        num_runs=10,
        use_flas=True,
    ),
]


def run_mode(runner: Task2Runner, cfg: ModeConfig, num_threads: int) -> Task2Result | None:
    print(f"\n{'='*60}")
    print(f"  Running: {cfg.label}")
    print(f"  Settings: {cfg.settings} | eps_search={cfg.eps_search}")
    print(f"{'='*60}\n")

    kwargs: dict = dict(
        mode=cfg.mode,
        k_ext=cfg.k_ext,
        k_graph=cfg.k_graph,
        eps_ext=cfg.eps_ext,
        threads=num_threads,
        build_threads=cfg.build_threads,  # Task 2: always build single-threaded
        max_dist=cfg.max_dist,
        eps_search=cfg.eps_search,
        num_runs=cfg.num_runs,
        use_flas=cfg.use_flas,
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


def find_best_point(sweep_points: list[dict[str, Any]], target_recall: float = 0.8) -> dict[str, Any] | None:
    """Find the point with search_time_ms minimized among those with recall >= target_recall."""
    valid_points = [p for p in sweep_points if p.get("recall", 0.0) >= target_recall]
    if valid_points:
        return min(valid_points, key=lambda p: p.get("search_time_ms", float("inf")))
    
    # Fallback: return point with highest recall if target not reached
    if sweep_points:
        return max(sweep_points, key=lambda p: p.get("recall", 0.0))
    return None


def generate_outputs(results: dict[str, Task2Result], output_dir: Path, system_info: dict | None = None) -> None:
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
    print(f"[benchmark_task2] Saved detailed JSON to {output_dir / 'results.json'}")

    # 2. Build plot
    plt.figure(figsize=(10, 6))
    
    # Custom color and style mapping for modes
    color_map = {
        "mode3": "tab:blue",
        "mode5": "tab:green",
        "mode4": "tab:orange",
        "mode6": "tab:red",
        "mode7": "tab:purple"
    }
    
    for cfg in MODES:
        res = results.get(cfg.name)
        if res is None or not res.sweep_points:
            continue
            
        # Extract points sorted by search time
        pts = sorted(res.sweep_points, key=lambda p: p.get("search_time_ms", 0.0))
        times = [p["search_time_ms"] for p in pts]
        recalls = [p["recall"] * 100.0 for p in pts]
        
        color = color_map.get(cfg.mode, "tab:gray")
        linestyle = "--" if not cfg.use_flas else "-"
        
        # Plot curve
        plt.plot(times, recalls, marker="o", label=cfg.label, linewidth=2, color=color, linestyle=linestyle)
        
        # Highlight best point
        best = find_best_point(res.sweep_points)
        if best:
            plt.scatter(
                [best["search_time_ms"]], [best["recall"] * 100.0],
                color=color, edgecolor="black", s=100, zorder=5
            )

    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    plt.axhline(y=80.0, color="gray", linestyle="--", label="Target Recall (80%)")
    plt.xlabel("Search Time (ms)")
    plt.ylabel("Recall @ 30 (%)")
    plt.title("Task 2 Benchmark — Recall @ 30 vs Search Time")
    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    
    plot_path = output_dir / "recall_vs_time.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[benchmark_task2] Saved plot to {plot_path}")

    # 3. Print & Save Markdown Summary
    md_content = []
    md_content.append("# Task 2 Benchmark Summary — Llama Dev dataset")
    md_content.append("\nThis table lists the best sweep configuration for each test that minimizes search time while reaching at least 80% recall.\n")
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
    
    headers = ["Mode", "Method", "Best Settings", "Load Time", "Build Time", "FLAS Time", "Total Time", "Search Time", "Recall"]
    md_content.append("| " + " | ".join(headers) + " |")
    md_content.append("|" + "|".join([":---:" if i == 0 or i > 2 else "---" for i in range(len(headers))]) + "|")
    
    for cfg in MODES:
        res = results.get(cfg.name)
        if res is None:
            md_content.append(f"| {cfg.mode} | {cfg.label} | ERR | — | — | — | — | — | — |")
            continue
            
        best = find_best_point(res.sweep_points)
        if best:
            best_settings = f"eps={best['eps_search']}, max_dist={best['max_dist']}"
            search_time_str = f"{best['search_time_ms']:.2f} ms"
            recall_str = f"{best['recall']*100.0:.2f}%"
        else:
            best_settings = "—"
            search_time_str = "—"
            recall_str = "—"
            
        load = f"{res.load_time_s:.1f}s" if res.load_time_s is not None else "—"
        build = f"{res.build_time_s:.1f}s" if res.build_time_s is not None else "—"
        flas = f"{res.flas_time_s:.1f}s" if res.flas_time_s is not None else "—"
        total = f"{res.overall_time_s:.1f}s" if res.overall_time_s is not None else "—"
        
        row = [cfg.mode, cfg.label, best_settings, load, build, flas, total, search_time_str, recall_str]
        md_content.append("| " + " | ".join(row) + " |")

    md_output = "\n".join(md_content)
    print("\n" + "=" * 60)
    print("  Benchmark Summary Table:")
    print("=" * 60)
    print(md_output)
    print("=" * 60 + "\n")
    
    with open(output_dir / "results.md", "w") as f:
        f.write(md_output)
    print(f"[benchmark_task2] Saved markdown summary to {output_dir / 'results.md'}")


def main() -> None:
    runner = Task2Runner(results_dir=Path(__file__).parent / "results", echo_logs=True)
    runner.build_image(force=False)
    system_info = runner.get_system_info()

    num_threads = runner.cpu_limit
    results: dict[str, Task2Result] = {}

    for cfg in MODES:
        sys.stdout.flush()
        result = run_mode(runner, cfg, num_threads)
        results[cfg.name] = result
        sys.stdout.flush()

    output_dir = Path(__file__).parent / "results" / "benchmark" / "task2"
    generate_outputs(results, output_dir, system_info)

    print("Done.")


if __name__ == "__main__":
    main()
