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
import cpuinfo
from utils.data import get_h5_file

DATASET_REPO = "sisap-challenges/SISAP2026"
SMALL_DATASET = True
ALGO_NAME = "deglib_evenregular_M24_LowLID"
TASK_NAME = "task1"
K = 15
EDGES_PER_VERTEX = 48

# Benchmark Results
# =================
#
# benchmark-dev-wikipedia-bge-m3-small.h5, k=15, EDGES_PER_VERTEX=16, LowLID:
# Edges/vertex:   16
# Build time:     15.2s
# Query time:     5.7s
# Throughput:     35354.5 q/s
# Recall:         0.4061
# Total time:     20.8s
#
# benchmark-dev-wikipedia-bge-m3-small.h5, k=15, EDGES_PER_VERTEX=24, LowLID:
# Build time:     25.8s
# Query time:     5.8s
# Throughput:     34647.9 q/s
# Recall:         0.5845
# Total time:     31.6s
#
# benchmark-dev-wikipedia-bge-m3-small.h5, k=15, EDGES_PER_VERTEX=32, LowLID:
# Build time:     41.3s
# Query time:     5.7s
# Throughput:     35123.1 q/s
# Recall:         0.6835
# Total time:     47.0s
#
# benchmark-dev-wikipedia-bge-m3-small.h5, k=15, EDGES_PER_VERTEX=48, LowLID:
# Build time:     76.6s
# Query time:     5.7s
# Throughput:     35075.8 q/s
# Recall:         0.7856
# Total time:     82.3s
#
# benchmark-dev-wikipedia-bge-m3.h5, k=15, EDGES_PER_VERTEX=48, LowLID:
# Build time:     1318.5s
# Query time:     114.1s
# Throughput:     55648.0 q/s
# Recall:         0.7140
# Total time:     1432.6s


def print_memory_usage(label: str = "Memory usage"):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{label}: {mem_info.rss / (1024 * 1024):.2f} MB")





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
    print_memory_usage("After graph build")

    print(f"Graph built: {g.size()} vertices, {EDGES_PER_VERTEX} edges/vertex")
    return g


def get_neighbors_k(
    g: graph.SizeBoundedGraph, n: int, k: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Retrieve k nearest neighbors for every vertex using direct edge weights."""
    start = time.perf_counter()
    all_indices: list[np.ndarray] = []
    all_dists: list[np.ndarray] = []

    for ext_label in range(n):
        internal_index = g.get_internal_index(ext_label)
        neighbors = g.get_neighbor_indices(internal_index)
        weights = g.get_neighbor_weights(internal_index)
        sorted_idx = np.argsort(weights)[:k]
        retrieved = np.array(
            [g.get_external_label(int(neighbors[j])) for j in sorted_idx]
        )
        dists = np.array([weights[j] for j in sorted_idx])
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
    # Convert 0-based to 1-based
    knns_1based = indices + 1
    dists_32 = dists.astype(np.float32)

    # Create results directory
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
        f.attrs["params"] = f"M={EDGES_PER_VERTEX},efConstruction=16,LowLID"

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
    path = get_h5_file(is_small=SMALL_DATASET)

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

    # Retrieve neighbors
    print(f"\n--- Retrieving {K} neighbors per vertex ---")
    indices, dists, qps = get_neighbors_k(g, g.size(), K)
    print(f"Query time: {indices.shape[0] / qps:.1f}s  QPS: {qps:.1f}")

    # Evaluate recall
    recall = evaluate_recall(indices, gold_allknn, K)
    print(f"Recall: {recall:.4f}")

    # Save results
    save_results(indices, dists, build_time, indices.shape[0] / qps, qps)

    # Summary
    cpu_info = cpuinfo.get_cpu_info()['brand_raw']
    ram_gb = psutil.virtual_memory().total / (1024**3)

    print("\n" + "="*45)
    print(f"{'SUMMARY':^45}")
    print("="*45)
    print(f"{'Dataset:':<25} {'Small' if SMALL_DATASET else 'Large'} ({g.size():,} elements)")
    print(f"{'Hardware:':<25} {cpu_info}")
    print(f"{'RAM:':<25} {ram_gb:.2f} GB")
    print("-" * 45)
    print(f"{'Settings:':<25} M={EDGES_PER_VERTEX}")
    print("-" * 45)
    print(f"{'Build time:':<25} {build_time:.1f} s")
    print(f"{'Query time:':<25} {indices.shape[0] / qps:.1f} s")
    print(f"{'Throughput:':<25} {qps:.1f} q/s")
    print(f"{'Recall:':<25} {recall:.4f}")
    print("-" * 45)
    print(f"{'Total time:':<25} {build_time + indices.shape[0] / qps:.1f} s")
    print("="*45)


if __name__ == "__main__":
    main()
