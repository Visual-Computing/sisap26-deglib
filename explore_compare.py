import deglib.builder as builder
import deglib.distances as dist
import deglib.graph as graph
import h5py
import numpy as np
from huggingface_hub import hf_hub_download

DATASET_REPO = "sisap-challenges/SISAP2026"
DATASET_FILE = "benchmark-dev-wikipedia-bge-m3-small.h5"
K = 15
EDGES_PER_VERTEX = 16


def download_dataset() -> str:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
    )
    print(f"Dataset downloaded to: {path}")
    return path


def main() -> None:
    path: str = download_dataset()

    with h5py.File(path, "r") as f:
        train: h5py.Dataset = f["train"]  # type: ignore[assignment]
        gold: h5py.Dataset = f["allknn"]["knns"]  # type: ignore[assignment]
        train = train[:].astype(np.float32)
        gold = gold[:].astype(np.int32) - 1  # type: ignore[operator]

    n: int = train.shape[0]
    g: graph.SizeBoundedGraph = graph.SizeBoundedGraph(
        n, EDGES_PER_VERTEX, dist.FloatSpace.create(1024, dist.Metric.InnerProduct)
    )
    b: builder.EvenRegularGraphBuilder = builder.EvenRegularGraphBuilder(
        g,
        optimization_target=builder.OptimizationTarget.LowLID,
        extend_k=EDGES_PER_VERTEX,
        extend_eps=0.001,
    )
    for i in range(n):
        b.add_entry(i, train[i].astype(np.float32))
    b.build()

    gold_0: np.ndarray = gold[0][:K]
    print("gold_allknn[0] erste 15 (externe Labels):")
    print(list(gold_0))

    rs = g.explore(0, K, False, EDGES_PER_VERTEX)
    explore_labels: list[int] = []
    print()
    print("explore(0) als externe Labels:")
    for item in rs.result_list:
        ext_label: int = g.get_external_label(item.get_internal_index())
        explore_labels.append(ext_label)
        dist_val: float = item.get_distance()
        print(
            f"  internal={item.get_internal_index()} -> external={ext_label}, dist={dist_val:.6f}"
        )

    overlap: set[int] = set(gold_0) & set(explore_labels)
    print()
    print(f"Overlap: {len(overlap)} / {min(len(gold_0), len(explore_labels))}")
    print(f"Overlap IDs: {sorted(overlap)}")


if __name__ == "__main__":
    main()
