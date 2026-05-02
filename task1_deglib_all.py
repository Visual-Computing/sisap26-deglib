import time
import os

import deglib.builder as builder
import deglib.distances as dist
import deglib.graph as graph
import h5py
import numpy as np
import psutil
import cpuinfo
from huggingface_hub import hf_hub_download
from utils.data import get_h5_file

DATASET_REPO = "sisap-challenges/SISAP2026"
SMALL_DATASET = True
K: int = 15
EDGES_PER_VERTEX: int = 16

# Graph build parameters
OPTIMIZATION_TARGET = "low_lid"

# Benchmark results (200K vertices, k=15, EDGES_PER_VERTEX=16, LowLID):
#   Method 1 (otest queries via g.search()):     Recall=0.4881  QPS=7429
#   Method 2 (graph neighbors sorted by weight):  Recall=0.4035  QPS=41842
#   Method 3 (graph explore from vertex ID):      Recall=0.5834  QPS=11801
#   Method 4 (graph search with vertex features): Recall=0.6660  QPS=6544
#
# Benchmark results (200K vertices, k=15, EDGES_PER_VERTEX=24, LowLID):
#   Method 1 (otest queries via g.search()):     Recall=0.6417  QPS=5761
#   Method 2 (graph neighbors sorted by weight):  Recall=0.5838  QPS=41780
#   Method 3 (graph explore from vertex ID):      Recall=0.7194  QPS=12058
#   Method 4 (graph search with vertex features): Recall=0.8315  QPS=5310





def build_graph(train_data: np.ndarray) -> graph.SizeBoundedGraph:
    n: int = train_data.shape[0]
    dim: int = train_data.shape[1]

    space: dist.FloatSpace = dist.FloatSpace.create(dim, dist.Metric.InnerProduct)
    g: graph.SizeBoundedGraph = graph.SizeBoundedGraph(n, EDGES_PER_VERTEX, space)

    target_map: dict[str, builder.OptimizationTarget] = {
        "low_lid": builder.OptimizationTarget.LowLID,
        "high_lid": builder.OptimizationTarget.HighLID,
        "streaming": builder.OptimizationTarget.StreamingData,
    }
    opt_target: builder.OptimizationTarget = target_map.get(
        OPTIMIZATION_TARGET, builder.OptimizationTarget.LowLID
    )

    b: builder.EvenRegularGraphBuilder = builder.EvenRegularGraphBuilder(
        g, optimization_target=opt_target, extend_k=EDGES_PER_VERTEX, extend_eps=0.001
    )
    for i in range(n):
        b.add_entry(i, train_data[i].astype(np.float32))
    b.build(callback="progress")

    print(f"Graph built: {g.size()} vertices, {EDGES_PER_VERTEX} edges/vertex")
    return g


def method_otest(
    g: graph.SizeBoundedGraph,
    queries: np.ndarray,
    gold_knns: np.ndarray,
    eps: float,
    max_dist_count: int,
) -> tuple[float, float]:
    """Search graph using otest query vectors via g.search(), compared against otest_knns."""
    n_queries: int = queries.shape[0]

    start: float = time.perf_counter()
    indices: np.ndarray = np.ndarray([])  # type: ignore[assignment]
    distances: np.ndarray = np.ndarray([])  # type: ignore[assignment]
    indices, distances = g.search(
        queries, eps, K, max_distance_computation_count=max_dist_count, threads=1
    )
    elapsed: float = time.perf_counter() - start

    qps: float = n_queries / elapsed

    correct: int = 0
    total: int = 0
    for q_idx in range(n_queries):
        gold_set: set[int] = set(gold_knns[q_idx][:K])
        retrieved_set: set[int] = set(indices[q_idx][:K])
        correct += len(gold_set & retrieved_set)
        total += K

    recall: float = correct / total if total > 0 else 0.0
    return recall, qps


def method_neighbors(
    g: graph.SizeBoundedGraph, gold_allknn: np.ndarray
) -> tuple[float, float, list[np.ndarray]]:
    """Use direct graph neighbors sorted by edge weight, compared against allknn."""
    n: int = g.size()

    start: float = time.perf_counter()
    all_retrieved: list[np.ndarray] = []
    for external_label in range(n):
        internal_index = g.get_internal_index(external_label)
        neighbors = g.get_neighbor_indices(internal_index)
        weights = g.get_neighbor_weights(internal_index)
        sorted_idx = np.argsort(weights)[:K]
        retrieved = np.array(
            [g.get_external_label(int(neighbors[j])) for j in sorted_idx]
        )
        all_retrieved.append(retrieved)
    elapsed: float = time.perf_counter() - start

    qps: float = n / elapsed

    correct: int = 0
    total: int = 0
    for i in range(n):
        gold_set: set[int] = set(gold_allknn[i][:K])
        retrieved_set: set[int] = set(all_retrieved[i][:K])
        correct += len(gold_set & retrieved_set)
        total += K

    recall: float = correct / total if total > 0 else 0.0
    return recall, qps, all_retrieved


def method_search_allknn(
    g: graph.SizeBoundedGraph,
    train_data: np.ndarray,
    gold_allknn: np.ndarray,
) -> tuple[float, float, list[list[int]]]:
    """Search graph using each graph vertex's own feature vector via g.search(), compared against allknn."""
    n: int = g.size()

    start: float = time.perf_counter()
    all_retrieved: list[list[int]] = []
    for i in range(n):
        query: np.ndarray = train_data[i].astype(np.float32)
        indices_i: np.ndarray = np.ndarray([])  # type: ignore[assignment]
        distances_i: np.ndarray = np.ndarray([])  # type: ignore[assignment]
        indices_i, distances_i = g.search(
            query, 0.001, K, max_distance_computation_count=0, threads=1
        )
        all_retrieved.append(list(indices_i[0]))
    elapsed: float = time.perf_counter() - start

    qps: float = n / elapsed

    correct: int = 0
    total: int = 0
    for i in range(n):
        gold_set: set[int] = set(gold_allknn[i][:K])
        retrieved_set: set[int] = set(all_retrieved[i][:K])
        correct += len(gold_set & retrieved_set)
        total += K

    recall: float = correct / total if total > 0 else 0.0
    return recall, qps, all_retrieved


def method_explore(
    g: graph.SizeBoundedGraph, gold_allknn: np.ndarray
) -> tuple[float, float, list[list[int]]]:
    """Explore graph from each graph vertex ID to find K nearest neighbors, compared against allknn."""
    n: int = g.size()

    start: float = time.perf_counter()
    all_retrieved: list[list[int]] = []
    for ext_label in range(n):
        internal_idx: int = g.get_internal_index(ext_label)
        rs = g.explore(internal_idx, K, False, 100)
        retrieved: list[int] = [
            g.get_external_label(item.get_internal_index()) for item in rs.result_list
        ]
        all_retrieved.append(retrieved)
    elapsed: float = time.perf_counter() - start

    qps: float = n / elapsed

    correct: int = 0
    total: int = 0
    for ext_label in range(n):
        gold_set: set[int] = set(gold_allknn[ext_label][:K])
        retrieved_set: set[int] = set(all_retrieved[ext_label][:K])
        correct += len(gold_set & retrieved_set)
        total += K

    recall: float = correct / total if total > 0 else 0.0
    return recall, qps, all_retrieved


def main() -> None:
    path: str = get_h5_file(is_small=SMALL_DATASET)

    with h5py.File(path, "r") as f:
        train_h5: h5py.Dataset = f["train"]  # type: ignore[assignment]
        gold_allknn_h5: h5py.Dataset = f["allknn"]["knns"]  # type: ignore[assignment]
        otest_queries_h5: h5py.Dataset = f["otest"]["queries"]  # type: ignore[assignment]
        otest_knns_h5: h5py.Dataset = f["otest"]["knns"]  # type: ignore[assignment]
        train: np.ndarray = train_h5[:]
        gold_allknn: np.ndarray = gold_allknn_h5[:]
        otest_queries: np.ndarray = otest_queries_h5[:].astype(np.float32)
        otest_knns: np.ndarray = otest_knns_h5[:]

    # gold_allknn is 1-based, convert to 0-based
    gold_allknn_0based: np.ndarray = gold_allknn.astype(np.int32) - 1  # type: ignore[operator]
    otest_knns_0based: np.ndarray = otest_knns.astype(np.int32) - 1  # type: ignore[operator]

    print(f"Train data shape: {train.shape}")
    print(f"Gold allknn shape: {gold_allknn.shape}")
    print(f"Otest queries shape: {otest_queries.shape}")
    print(f"Otest knns shape: {otest_knns.shape}")

    print("\n--- Building graph ---")
    graph_start: float = time.perf_counter()
    g: graph.SizeBoundedGraph = build_graph(train)
    build_time: float = time.perf_counter() - graph_start
    print(f"Build time: {build_time:.1f}s")

    print(f"\n--- Method 1: otest queries via g.search() (k={K}) ---")
    for eps in [0.001]:
        recall1, qps1 = method_otest(g, otest_queries, otest_knns_0based, eps, 0)
        print(f"  eps={eps:>7.3f}: Recall={recall1:.4f}  QPS={qps1:.1f}")

    print(f"\n--- Method 2: graph neighbors sorted by weight (k={K}) ---")
    recall2, qps2, retrieved2 = method_neighbors(g, gold_allknn_0based)
    print(f"  Recall: {recall2:.4f}")
    print(f"  QPS:    {qps2:.1f}")

    print(f"\n--- Method 3: graph explore from graph vertex ID (k={K}) ---")
    recall3, qps3, retrieved3 = method_explore(g, gold_allknn_0based)
    print(f"  Recall={recall3:.4f}  QPS={qps3:.1f}")

    print(f"\n--- Method 4: graph search with vertex features (k={K}) ---")
    recall4, qps4, retrieved4 = method_search_allknn(g, train, gold_allknn_0based)
    print(f"  Recall={recall4:.4f}  QPS={qps4:.1f}")

    # Summary
    cpu_info = cpuinfo.get_cpu_info()['brand_raw']
    ram_gb = psutil.virtual_memory().total / (1024**3)

    print("\n" + "="*45)
    print(f"{'SUMMARY':^45}")
    print("="*45)
    print(f"{'Dataset:':<25} {'Small' if SMALL_DATASET else 'Large'} ({train.shape[0]:,} elements)")
    print(f"{'Hardware:':<25} {cpu_info}")
    print(f"{'RAM:':<25} {ram_gb:.2f} GB")
    print("-" * 45)
    print(f"{'Settings:':<25} M={EDGES_PER_VERTEX}, Opt={OPTIMIZATION_TARGET}")
    print("-" * 45)
    print(f"{'Method 1 Recall:':<25} {recall1:.4f}")
    print(f"{'Method 1 QPS:':<25} {qps1:.1f}")
    print(f"{'Method 2 Recall:':<25} {recall2:.4f}")
    print(f"{'Method 2 QPS:':<25} {qps2:.1f}")
    print(f"{'Method 3 Recall:':<25} {recall3:.4f}")
    print(f"{'Method 3 QPS:':<25} {qps3:.1f}")
    print(f"{'Method 4 Recall:':<25} {recall4:.4f}")
    print(f"{'Method 4 QPS:':<25} {qps4:.1f}")
    print("-" * 45)
    print(f"{'Build time:':<25} {build_time:.1f} s")
    print("="*45)


if __name__ == "__main__":
    main()
