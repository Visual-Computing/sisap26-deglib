import time

import deglib.builder as builder
import deglib.distances as dist
import deglib.graph as graph
import h5py
import numpy as np
from huggingface_hub import hf_hub_download

DATASET_REPO = "sisap-challenges/SISAP2026"
DATASET_FILE = "benchmark-dev-wikipedia-bge-m3-small.h5"
ALGO_NAME = "deglib_evenregular_M24_LowLID_explore"
TASK_NAME = "task1"
K = 15
EDGES_PER_VERTEX = 32
MAX_EXPLORE_DISTANCES = 100

# Benchmark results (200K vertices, k=15, EDGES_PER_VERTEX=32, MAX_EXPLORE_DISTANCES=100, LowLID, explore):
# Build time:     39.9s
# Query time:     18.0s
# Throughput:     11082.5 q/s
# Recall:         0.7810
# Total time:     58.0s


def download_dataset() -> str:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
    )
    print(f"Dataset downloaded to: {path}")
    return path


def build_graph(train_data: np.ndarray) -> graph.SizeBoundedGraph:
    n = train_data.shape[0]
    dim = train_data.shape[1]

    space = dist.FloatSpace.create(dim, dist.Metric.InnerProduct)
    g = graph.SizeBoundedGraph(n, EDGES_PER_VERTEX, space)

    b = builder.EvenRegularGraphBuilder(
        g,
        optimization_target=builder.OptimizationTarget.LowLID,
        extend_k=EDGES_PER_VERTEX,
        extend_eps=0.001,
    )
    for i in range(n):
        b.add_entry(i, train_data[i].astype(np.float32))
    b.build(callback="progress")

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

    with h5py.File(path, "r") as f:  # type: ignore[assignment]
        f: h5py.File  # type: ignore[misc]
        train = np.array(f["train"], dtype=np.float32)  # type: ignore[index]
        gold_allknn_raw = np.array(f["allknn"]["knns"], dtype=np.int32)  # type: ignore[index]
        gold_allknn = gold_allknn_raw - 1  # type: ignore[operator]

    print(f"Train data shape: {train.shape}")
    print(f"Gold allknn shape: {gold_allknn.shape}")
    print(f"Edges per vertex: {EDGES_PER_VERTEX}, k: {K}")

    # Build graph
    print("\n--- Building graph ---")
    graph_start = time.perf_counter()
    g = build_graph(train)
    build_time = time.perf_counter() - graph_start
    print(f"Build time: {build_time:.1f}s")

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
