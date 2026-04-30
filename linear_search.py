import time
from typing import Any

import h5py
import numpy as np
from huggingface_hub import hf_hub_download

DATASET_REPO = "sisap-challenges/SISAP2026"
DATASET_FILE = "benchmark-dev-wikipedia-bge-m3-small.h5"
K: int = 15


def download_dataset() -> str:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
    )
    print(f"Dataset downloaded to: {path}")
    return path


def linear_search_topk(
    train: np.ndarray, query_idx: int, k: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Find k nearest neighbors using linear scan (dot product)."""
    query: np.ndarray = train[query_idx]
    start: float = time.perf_counter()
    scores: np.ndarray = train @ query
    elapsed: float = time.perf_counter() - start

    topk_idx: np.ndarray = np.argsort(scores)[-k:][::-1]
    topk_scores: np.ndarray = scores[topk_idx]

    qps: float = 1 / elapsed
    return topk_idx, topk_scores, qps, elapsed


def compare_with_gold(
    topk_idx: np.ndarray, gold_allknn: np.ndarray, query_idx: int
) -> tuple[float, int]:
    """Compare topk results against gold standard."""
    gold_set: set[int] = set(gold_allknn[query_idx][:K])
    retrieved_set: set[int] = set(topk_idx)
    overlap: int = len(gold_set & retrieved_set)
    recall: float = overlap / K
    return recall, overlap


def main() -> None:
    path: str = download_dataset()

    with h5py.File(path, "r") as f:
        train: h5py.Dataset = f["train"]  # type: ignore[assignment]
        gold_allknn: h5py.Dataset = f["allknn"]["knns"]  # type: ignore[assignment]
        train = train[:].astype(np.float32)
        gold_allknn = gold_allknn[:].astype(np.int32) - 1  # type: ignore[operator]

    print(f"Train data shape: {train.shape}")
    print(f"Gold allknn shape: {gold_allknn.shape}")

    query_idx: int = 0
    print(f"\n--- Linear search for graph vertex {query_idx} (k={K}) ---")

    topk_idx, topk_scores, qps, elapsed = linear_search_topk(train, query_idx, K)

    print(f"Time: {elapsed:.4f}s")
    print(f"Queries/sec: {qps:.1f}")
    print()
    print("Top 15 (Index, Score):")
    for idx, sc in zip(topk_idx, topk_scores):
        print(f"  {int(idx):>6}  {sc:.6f}")
    print()
    print(f"Gold allknn[{query_idx}] erste {K}:")
    for idx in gold_allknn[query_idx][:K]:
        print(f"  {int(idx):>6}")
    print()

    recall, overlap = compare_with_gold(topk_idx, gold_allknn, query_idx)
    print(f"Overlap: {overlap} / {K}")
    print(f"Recall: {recall:.4f}")

    if overlap == K:
        print("\nPerfekt! Alle Top 15 stimmen mit Gold ueberein.")
    else:
        missing: set[int] = set(gold_allknn[query_idx][:K]) - set(topk_idx)
        extra: set[int] = set(topk_idx) - set(gold_allknn[query_idx][:K])
        if missing:
            print(f"Fehlende IDs: {sorted(missing)}")
        if extra:
            print(f"Zusaetzliche IDs: {sorted(extra)}")


if __name__ == "__main__":
    main()
