import time
import sys
import json
from pathlib import Path
from enum import Enum
import h5py
import numpy as np
from huggingface_hub import hf_hub_download
import deglib
import flas_cpp


class SortOrder(Enum):
    ORIGINAL = "original"
    SHUFFLE = "shuffle"
    FLAS = "flas"


# ==============================================================================
# BENCHMARK CONFIGURATION
# ==============================================================================
# Dataset parameters
REPO_ID = "SISAP-Challenges/SISAP2026"
FILENAME = "llama-dev/llama-dev.h5"

# Input options: Set LOAD_GRAPH = True to load a pre-built graph instead of building a new one
LOAD_GRAPH = False
LOAD_GRAPH_PATH = str(Path(__file__).parent / "results/llama_dev_deg.graph")

# Sorting order for train vectors: SortOrder.ORIGINAL, SortOrder.SHUFFLE, SortOrder.FLAS
# (Ignored if LOAD_GRAPH is True)
SORT_ORDER = SortOrder.SHUFFLE

# Shuffling parameters (used if SORT_ORDER == SortOrder.SHUFFLE)
SHUFFLE_SEED = 42

# FLAS sorting parameters (used if SORT_ORDER == SortOrder.FLAS)
FLAS_SEED = 42
FLAS_RADIUS_DECAY = 0.93
FLAS_MAX_SWAP_POSITIONS = 9
FLAS_OPTIMIZE_NARROW_GRIDS = 1

# DEG Graph Build Parameters
# (Ignored if LOAD_GRAPH is True)
K_GRAPH = 30
K_EXT = 30
EPS_EXT = 0.001
IMPROVE_K = 0
IMPROVE_EPS = 0.0
MAX_PATH_LENGTH = 5
SWAP_TRIES = 0
ADDITIONAL_SWAP_TRIES = 0
OPTIMIZATION_TARGET = deglib.builder.OptimizationTarget.LowLID

# Thread limits
BUILD_THREADS = 1
SEARCH_THREADS = 8

# Search Evaluation Parameters
K_SEARCH = 30
EPS_SWEEP = [0.1, 0.15, 0.2, 0.25, 0.3]

# Output options
SAVE_GRAPH = False
SAVE_GRAPH_PATH = str(Path(__file__).parent / "results/llama_dev_deg.graph")
# ==============================================================================


def flas_progress_callback(progress: float) -> bool:
    bar_length = 60
    block = int(bar_length * progress)
    bar = '#' * block + '-' * (bar_length - block)
    percentage = progress * 100
    sys.stdout.write(f'\rFLAS Sorting: {percentage:6.2f}% [{bar}]')
    if progress >= 1.0:
        sys.stdout.write('\n')
    sys.stdout.flush()
    return False


def main():
    print("=" * 60)
    print("  deglib - Task 2 Direct Benchmark")
    print("=" * 60)

    # 1. Download/resolve the dataset using huggingface_hub
    print(f"\n[1/5] Downloading/resolving {FILENAME} dataset from Hugging Face...")
    t_download_start = time.perf_counter()
    h5_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
    )
    t_download_end = time.perf_counter()
    print(f"      -> Path: {h5_path}")
    print(f"      -> Resolved in {t_download_end - t_download_start:.2f} s")

    # 2. Load queries and ground truth (and train only if building graph)
    print("\n[2/5] Loading data from HDF5 file...")
    t_load_start = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        queries = np.array(f["test/queries"], dtype=np.float32)
        knns = np.array(f["test/knns"], dtype=np.int64)
        if not LOAD_GRAPH:
            train = np.array(f["train"], dtype=np.float32)
            train_loaded_msg = f"Loaded {train.shape[0]} train vectors ({train.shape[1]}-dim)"
        else:
            train = None
            train_loaded_msg = "Skipped loading train vectors (using pre-built graph)"
    t_load_end = time.perf_counter()
    print(f"      -> {train_loaded_msg}")
    print(f"      -> Loaded {queries.shape[0]} query vectors")
    print(f"      -> Loaded ground truth knns shape: {knns.shape}")
    print(f"      -> Time elapsed: {t_load_end - t_load_start:.2f} s")

    # 3. Build or Load the DEG Graph
    if LOAD_GRAPH:
        print(f"\n[3/5] Loading DEG Graph from {LOAD_GRAPH_PATH}...")
        if not Path(LOAD_GRAPH_PATH).exists():
            raise FileNotFoundError(f"Saved graph not found at: {LOAD_GRAPH_PATH}")
        t_graph_load_start = time.perf_counter()
        graph = deglib.graph.load_readonly_graph(LOAD_GRAPH_PATH)
        t_graph_load_end = time.perf_counter()
        build_time = t_graph_load_end - t_graph_load_start
        print(f"      -> DEG Graph loaded successfully in {build_time:.2f} s")
        print(f"      -> Graph size: {graph.size()} vertices")
    else:
        print(f"\n[3/5] Building DEG Graph (threads={BUILD_THREADS})...")
        n_train = train.shape[0]
        dims = train.shape[1]

        # Create empty graph
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            capacity=n_train,
            dims=dims,
            edges_per_vertex=K_GRAPH,
            metric=deglib.Metric.InnerProduct
        )

        # Setup 1-based labels to match SISAP ground truth
        labels = np.arange(1, n_train + 1, dtype=np.uint32)

        # Apply sorting order to train vectors and labels
        if SORT_ORDER == SortOrder.SHUFFLE:
            print(f"      -> Shuffling train vectors (seed={SHUFFLE_SEED})...")
            rng = np.random.default_rng(seed=SHUFFLE_SEED)
            perm = rng.permutation(n_train)
            train = train[perm]
            labels = labels[perm]
        elif SORT_ORDER == SortOrder.FLAS:
            print("      -> Sorting train vectors with FLAS (1 x N grid)...")
            t_flas_start = time.perf_counter()
            # 1 x N grid represented as (N, 1) shape
            ids = np.arange(n_train, dtype=np.int32).reshape(n_train, 1)
            frozen = np.zeros((n_train, 1), dtype=bool)
            
            code, result = flas_cpp.flas(
                train.astype(np.float32),
                ids,
                frozen,
                False,                       # wrap
                FLAS_RADIUS_DECAY,           # radius_decay
                1.0,                         # weight_swappable
                100.0,                       # weight_non_swappable
                0.01,                        # weight_hole
                FLAS_MAX_SWAP_POSITIONS,     # max_swap_positions
                FLAS_OPTIMIZE_NARROW_GRIDS,  # optimize_narrow_grids
                FLAS_SEED,                   # seed
                flas_progress_callback       # callback
            )
            if code != 0:
                raise RuntimeError(f"FLAS C++ extension failed with code {code}")
            t_flas_end = time.perf_counter()
            print(f"      -> FLAS sorting completed in {t_flas_end - t_flas_start:.2f} s")
            
            perm = result.flatten()
            train = train[perm]
            labels = labels[perm]
        elif SORT_ORDER == SortOrder.ORIGINAL:
            print("      -> Keeping original train vector order...")
        else:
            raise ValueError(f"Unknown SORT_ORDER: {SORT_ORDER}. Expected 'original', 'shuffle', or 'flas'.")

        # Configure builder
        builder = deglib.builder.EvenRegularGraphBuilder(
            graph,
            optimization_target=OPTIMIZATION_TARGET,
            extend_k=K_EXT,
            extend_eps=EPS_EXT,
            improve_k=IMPROVE_K,
            improve_eps=IMPROVE_EPS,
            max_path_length=MAX_PATH_LENGTH,
            swap_tries=SWAP_TRIES,
            additional_swap_tries=ADDITIONAL_SWAP_TRIES,
        )
        builder.set_thread_count(BUILD_THREADS)
        builder.add_entry(labels, train)

        # Build and measure time
        t_build_start = time.perf_counter()
        builder.build(callback="progress")
        t_build_end = time.perf_counter()
        build_time = t_build_end - t_build_start
        print(f"      -> DEG Graph built successfully in {build_time:.2f} s")
        print(f"      -> Graph size: {graph.size()} vertices")

    # 4. Search and Sweep eps-search
    print(f"\n[4/5] Running search sweep (eps-search {EPS_SWEEP[0]} to {EPS_SWEEP[-1]}, threads={SEARCH_THREADS})...")
    
    results = []
    
    # We want Recall@K_SEARCH, so extract first K_SEARCH columns of ground truth knns
    gt_k = knns[:, :K_SEARCH]

    for eps in EPS_SWEEP:
        t_search_start = time.perf_counter()
        # Search the top K_SEARCH closest neighbors
        indices, _ = graph.search(
            query=queries,
            eps=eps,
            k=K_SEARCH,
            threads=SEARCH_THREADS
        )
        t_search_end = time.perf_counter()
        
        search_time_s = t_search_end - t_search_start
        search_time_ms = search_time_s * 1000.0
        avg_time_ms = search_time_ms / len(queries)
        qps = len(queries) / search_time_s

        # Calculate Recall@K_SEARCH
        recall_sum = 0.0
        for i in range(len(queries)):
            retrieved = set(indices[i])
            ground_truth = set(gt_k[i])
            intersection = retrieved.intersection(ground_truth)
            recall_sum += len(intersection) / float(K_SEARCH)
        mean_recall = recall_sum / len(queries)

        results.append({
            "eps": eps,
            "recall": mean_recall,
            "search_time_ms": search_time_ms,
            "avg_time_ms": avg_time_ms,
            "qps": qps
        })
        print(f"      -> eps={eps:<5} | Recall@{K_SEARCH}={mean_recall:.2%} | Search Time={search_time_ms:.1f} ms | Avg={avg_time_ms:.3f} ms | QPS={qps:.1f}")

    # 5. Output Results Table
    print(f"\n[5/5] Final Benchmark Results Table:")
    print("=" * 70)
    print(f"{'Load' if LOAD_GRAPH else 'Build'} Time: {build_time:.2f} s")
    print(f"Sort Order: {SORT_ORDER.name if not LOAD_GRAPH else 'N/A (Pre-built)'}")
    print("=" * 70)
    print(f"| {'eps-search':<10} | {f'Recall@{K_SEARCH}':<10} | {'Search Time':<12} | {'Avg/Query':<10} | {'QPS':<10} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*12}|")
    for r in results:
        print(f"| {r['eps']:<10.3f} | {r['recall']:.2%} | {r['search_time_ms']:>8.1f} ms | {r['avg_time_ms']:>7.3f} ms | {r['qps']:>8.1f} |")
    print("=" * 70)

    # 6. Save graph and parameters metadata
    if SAVE_GRAPH and not LOAD_GRAPH:
        print(f"\nSaving DEG Graph and parameters metadata...")
        # Create parent directories if they do not exist
        Path(SAVE_GRAPH_PATH).parent.mkdir(parents=True, exist_ok=True)
        # Save the graph
        graph.save_graph(SAVE_GRAPH_PATH)
        print(f"      -> Saved graph file to: {SAVE_GRAPH_PATH}")
        
        # Save the metadata next to it
        metadata_path = Path(SAVE_GRAPH_PATH).with_suffix(".json")
        metadata = {
            "dataset": {
                "repo_id": REPO_ID,
                "filename": FILENAME
            },
            "sort_order": SORT_ORDER.name,
            "shuffle_seed": SHUFFLE_SEED,
            "flas": {
                "seed": FLAS_SEED,
                "radius_decay": FLAS_RADIUS_DECAY,
                "max_swap_positions": FLAS_MAX_SWAP_POSITIONS,
                "optimize_narrow_grids": FLAS_OPTIMIZE_NARROW_GRIDS
            },
            "graph": {
                "k_graph": K_GRAPH,
                "k_ext": K_EXT,
                "eps_ext": EPS_EXT,
                "improve_k": IMPROVE_K,
                "improve_eps": IMPROVE_EPS,
                "max_path_length": MAX_PATH_LENGTH,
                "swap_tries": SWAP_TRIES,
                "additional_swap_tries": ADDITIONAL_SWAP_TRIES,
                "optimization_target": OPTIMIZATION_TARGET.name,
                "build_threads": BUILD_THREADS,
                "build_time_s": build_time,
                "size": graph.size()
            },
            "search": {
                "k_search": K_SEARCH,
                "eps_sweep": EPS_SWEEP,
                "search_threads": SEARCH_THREADS
            },
            "results": [
                {
                    "eps": r["eps"],
                    "recall": r["recall"],
                    "search_time_ms": r["search_time_ms"],
                    "avg_time_ms": r["avg_time_ms"],
                    "qps": r["qps"]
                }
                for r in results
            ]
        }
        with open(metadata_path, "w") as f_meta:
            json.dump(metadata, f_meta, indent=4)
        print(f"      -> Saved metadata parameters to: {metadata_path}")

    return results


if __name__ == "__main__":
    main()
