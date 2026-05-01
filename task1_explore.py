import os
import sys
import time
from typing import cast

import deglib.builder as builder
import deglib.distances as dist
import deglib.graph as graph
import h5py
import numpy as np
import psutil
from huggingface_hub import hf_hub_download

DATASET_REPO = "sisap-challenges/SISAP2026"
# DATASET_FILE = "benchmark-dev-wikipedia-bge-m3-small.h5"  # small
DATASET_FILE = "benchmark-dev-wikipedia-bge-m3.h5"  # large
ALGO_NAME = "deglib_evenregular_M24_LowLID_explore"
TASK_NAME = "task1"
K = 15
EDGES_PER_VERTEX = 32
MAX_EXPLORE_DISTANCES = 100

# Benchmark Results
# =================
#
# benchmark-dev-wikipedia-bge-m3-small.h5, k=15, EDGES_PER_VERTEX=32, MAX_EXPLORE_DISTANCES=100, LowLID:
# Build time:     39.9s
# Query time:     18.0s
# Throughput:     11082.5 q/s
# Recall:         0.7810
# Total time:     58.0s
#
# benchmark-dev-wikipedia-bge-m3.h5, k=15, EDGES_PER_VERTEX=32, MAX_EXPLORE_DISTANCES=100, LowLID:
# Build time:     692.8s
# Query time:     358.5s
# Throughput:     17713.3 q/s
# Recall:         0.7044
# Total time:     1051.3s


def print_memory_usage(label: str = "Memory usage"):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{label}: {mem_info.rss / (1024 * 1024):.2f} MB")


def download_dataset() -> str:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
    )
    print(f"Dataset downloaded to: {path}")
    return path


def build_graph(train_data) -> graph.SizeBoundedGraph:
    n = train_data.shape[0]
    dim = train_data.shape[1]

    print_memory_usage("Initial memory")

    space = dist.FloatSpace.create(dim, dist.Metric.InnerProduct)
    g = graph.SizeBoundedGraph(n, EDGES_PER_VERTEX, space)

    print_memory_usage("After graph allocation")

    b = builder.EvenRegularGraphBuilder(
        g,
        optimization_target=builder.OptimizationTarget.LowLID,
        extend_k=EDGES_PER_VERTEX,
        extend_eps=0.001,
    )

    batch_size = 10000
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_features = train_data[i:end].astype(np.float32)
        batch_labels = np.arange(i, end, dtype=np.uint32)
        b.add_entry(batch_labels, batch_features)
        b.build()

        progress = end / n
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "#" * filled + "-" * (bar_length - filled)
        sys.stdout.write(f"\rBuilding graph: [{bar}] {progress * 100:.1f}% ({end}/{n})")
        sys.stdout.flush()
    print()
    print_memory_usage("After graph build")

    print(f"Graph built: {g.size()} vertices, {EDGES_PER_VERTEX} edges/vertex")
    return g


def explore_k(
    g: graph.SizeBoundedGraph, n: int, k: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Retrieve k nearest neighbors for every vertex using g.explore()."""
    start = time.perf_counter()
    all_indices: list[np.ndarray] = []
    all_dists: list[np.ndarray] = []

    for ext_label in range(n):
        internal_idx = g.get_internal_index(ext_label)
        rs = g.explore(internal_idx, k, False, MAX_EXPLORE_DISTANCES)
        retrieved = np.array(
            [g.get_external_label(item.get_internal_index()) for item in rs.result_list]
        )
        dists = np.array([item.get_distance() for item in rs.result_list])
        all_indices.append(retrieved)
        all_dists.append(dists)

    elapsed = time.perf_counter() - start
    qps = n / elapsed

    indices_arr = np.array(all_indices)
    dists_arr = np.array(all_dists)
    return indices_arr, dists_arr, qps


def save_results(
    indices: np.ndarray,
    dists: np.ndarray,
    build_time: float,
    query_time: float,
    qps: float,
) -> str:
    """Save results in the SISAP 2026 submission HDF5 format (1-based indexing)."""
    knns_1based = indices + 1
    dists_32 = dists.astype(np.float32)

    import os

    os.makedirs("results/task1", exist_ok=True)

    filename = (
        f"results/task1/{ALGO_NAME.replace(' ', '_')}_M{EDGES_PER_VERTEX}_k{K}.h5"
    )
    with h5py.File(filename, "w") as f:
        f.create_dataset("knns", data=knns_1based)
        f.create_dataset("dists", data=dists_32)
        f.attrs["algo"] = ALGO_NAME
        f.attrs["task"] = TASK_NAME
        f.attrs["buildtime"] = float(build_time)
        f.attrs["querytime"] = float(query_time)
        f.attrs["params"] = f"M={EDGES_PER_VERTEX},efConstruction=16,LowLID,explore"

    print(f"Results saved to: {filename}")
    return filename


def evaluate_recall(indices: np.ndarray, gold_allknn: np.ndarray, k: int) -> float:
    """Calculate recall against gold standard."""
    correct = 0
    total = 0
    for i in range(indices.shape[0]):
        gold_set = set(gold_allknn[i][:k])
        retrieved_set = set(indices[i][:k])
        correct += len(gold_set & retrieved_set)
        total += k
    return correct / total if total > 0 else 0.0


def main() -> None:
    path = download_dataset()

    with h5py.File(path, "r") as f:
        # Train data is used lazily through the HDF5 dataset object
        train_data = cast(h5py.Dataset, f["train"])

        print(f"Train data shape: {train_data.shape}")
        print(f"Edges per vertex: {EDGES_PER_VERTEX}, k: {K}")

        # Build graph
        print("\n--- Building graph ---")
        graph_start = time.perf_counter()
        g = build_graph(train_data)
        build_time = time.perf_counter() - graph_start
        print(f"Build time: {build_time:.1f}s")

    # Now load gold allknn after the graph is built to save memory
    with h5py.File(path, "r") as f:
        allknn_group = cast(h5py.Group, f["allknn"])
        gold_allknn_raw = np.array(allknn_group["knns"], dtype=np.int32)
        gold_allknn = gold_allknn_raw - 1

    # Retrieve neighbors via explore
    print(f"\n--- Retrieving {K} neighbors per vertex via g.explore() ---")
    indices, dists, qps = explore_k(g, g.size(), K)
    print(f"Query time: {indices.shape[0] / qps:.1f}s  QPS: {qps:.1f}")

    # Evaluate recall
    recall = evaluate_recall(indices, gold_allknn, K)
    print(f"Recall: {recall:.4f}")

    # Save results
    save_results(indices, dists, build_time, indices.shape[0] / qps, qps)

    # Summary
    print("\n=== Summary ===")
    print(f"Algorithm:      {ALGO_NAME}")
    print(f"Edges/vertex:   {EDGES_PER_VERTEX}")
    print(f"Build time:     {build_time:.1f}s")
    print(f"Query time:     {indices.shape[0] / qps:.1f}s")
    print(f"Throughput:     {qps:.1f} q/s")
    print(f"Recall:         {recall:.4f}")
    print(f"Total time:     {build_time + indices.shape[0] / qps:.1f}s")


if __name__ == "__main__":
    main()
