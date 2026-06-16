import argparse
import h5py
import numpy as np
import csv
import glob
from datasets import DATASETS, get_fn, prepare, get_h5_item


def get_all_results(dirname):
    """Yield (path, attrs, knns) for every valid result .h5 file under dirname."""
    seen = set()
    mask = dirname + "/**/*.h5"
    print(f"Searching for results matching: {mask}")
    for fn in glob.iglob(mask, recursive=True):
        if fn in seen:
            continue
        seen.add(fn)
        with h5py.File(fn, "r") as f:
            if "knns" not in f or "dataset" not in f.attrs or "task" not in f.attrs:
                print(f"Ignoring {fn}")
                continue
            print(fn)
            yield fn, dict(f.attrs), np.array(f["knns"])


def get_recall(I, gt, k):
    """Compute Recall@k averaged over all queries."""
    assert k <= I.shape[1], f"k={k} exceeds result columns {I.shape[1]}"
    assert I.shape[0] == gt.shape[0], "query count mismatch between results and ground truth"
    hits = np.array([
        len(np.intersect1d(I[i, :k], gt[i, :k]))
        for i in range(len(I))
    ])
    return hits.sum() / (len(I) * k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        help="directory in which results are stored",
        default="results",
    )
    parser.add_argument("csvfile")
    args = parser.parse_args()

    gt_cache = {}  # (dataset, task) -> gt_I array
    columns = ["dataset", "task", "algo", "buildtime", "querytime", "params", "recall"]

    with open(args.csvfile, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for fn, attrs, knns in get_all_results(args.results):
            dataset = attrs["dataset"]
            task = attrs["task"]

            if dataset not in DATASETS or task not in DATASETS[dataset]:
                print(f"Skipping {fn}: unknown dataset={dataset!r} task={task!r}")
                continue

            prepare(dataset, task)

            cache_key = (dataset, task)
            if cache_key not in gt_cache:
                gt_fn = get_fn(dataset, task)
                print(f"Loading ground truth from {gt_fn}")
                with h5py.File(gt_fn) as gf:
                    gt_I = np.array(get_h5_item(gf, DATASETS[dataset][task]["gt_I"]))
                gt_cache[cache_key] = gt_I

            gt_I = gt_cache[cache_key]
            k = DATASETS[dataset][task]["k"]
            recall = get_recall(knns, gt_I, k)
            row = dict(attrs)
            row["recall"] = recall
            print(dataset, task, attrs.get("algo"), attrs.get("params"), "=>", recall)
            writer.writerow(row)
