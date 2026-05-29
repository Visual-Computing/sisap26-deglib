"""
main.py — End-to-end example for running deglib_evp_task1 via Docker.

Usage
-----
    uv run python main.py

What this does
--------------
1. Connects to the local Docker daemon.
2. Downloads the SISAP 2026 small dataset from HuggingFace (cached after
   the first run — subsequent runs start instantly).
3. Starts the sisap26-deglib container with SISAP-compliant resource
   limits (8 CPUs, 24 GB RAM, no swap).
4. Streams container logs to stdout in real time.
5. Prints a structured summary of timing and recall results.

Prerequisites
-------------
- Docker Desktop must be running.
- The Docker image must be built first:
      docker build -t sisap26-deglib .
"""
from pathlib import Path

from docker_runner import Task1Runner


def main() -> None:
    runner = Task1Runner(
        image_tag="sisap26-deglib",
        results_dir=Path("./results"),
        echo_logs=True,   # print every container log line as it arrives
    )

    runner.build_image(force=False)

    print("=" * 60)
    print("  deglib EVP Task 1 — Docker Runner")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    # The runner downloads the dataset from HuggingFace on first call and
    # caches it locally. Subsequent calls return the cached path instantly.
    size = "small"   # switch to "large" for the full 6.35 M vector dataset
    print(f"\n[1/2] Resolving {size!r} dataset …")
    data_path = runner.get_dataset_path(size=size)
    print(f"      → {data_path}")

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #
    print(f"\n[2/2] Running container (mode=mode4, max_dist sweep 100,200,300) …\n")

    result = runner.run(
        mode="mode4",                  # EVP build + EVP explore + FP16 rerank
        size=size,
        threads=8,                     # use all 8 allocated CPUs
        max_dist="100,200,300",        # comma-separated → sweep over 3 budgets
        evp_k=50,
    )

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)

    if not result.succeeded:
        print(f"ERROR: container exited with code {result.exit_code}")
        return

    print(f"  SIMD          : {result.simd_info or 'n/a'}")
    print(f"  Load          : {_fmt(result.load_time_s)}")
    print(f"  Quantisation  : {_fmt(result.quant_time_s)}")
    print(f"  Build         : {_fmt(result.build_time_s)}")
    print(f"  Convert       : {_fmt(result.convert_time_s)}")
    print(f"  Explore       : {_fmt(result.explore_time_s)}")
    print(f"  Rerank        : {_fmt(result.rerank_time_s)}")
    print(f"  Total         : {_fmt(result.overall_time_s)}")
    print()

    if result.recall_results:
        print("  Recall sweep:")
        for max_dist, recall in result.recall_results:
            bar = "⭐" if recall >= 0.80 else "  "
            print(f"    max_dist={max_dist:>4}  →  Recall@15 = {recall:.4f}  {bar}")
    else:
        print("  No recall results (was --no-recall set?)")

    print()
    print(f"  Best recall   : {result.best_recall:.4f}" if result.best_recall else "  Best recall   : n/a")
    print("=" * 60)


def _fmt(value: float | None) -> str:
    return f"{value:.2f} s" if value is not None else "n/a"


if __name__ == "__main__":
    main()
