import h5py
import json
import os
from urllib.request import urlretrieve
from pathlib import Path
from scipy.sparse import csr_matrix
from glob import glob


def _load_config(dataset):
    """Load and return the config.json for a dataset as a dict."""
    cfg_path = os.path.join("data", dataset, "config.json")
    with open(cfg_path) as f:
        return json.load(f)


def _discover_datasets():
    """Scan data/ for config.json files and return a nested dict keyed by
    dataset_name then task name, mirroring the structure eval.py expects:
    DATASETS[dataset_name][task] -> config dict."""
    datasets = {}
    for cfg_path in sorted(glob("data/*/config.json")):
        cfg = json.loads(Path(cfg_path).read_text())
        dataset_name = cfg.get("dataset_name", Path(cfg_path).parent.name)
        task = cfg.get("task")
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name][task] = cfg
        # Store the folder name so get_fn can locate the file
        datasets[dataset_name][task]["_folder"] = Path(cfg_path).parent.name
    return datasets


DATASETS = _discover_datasets()


def get_fn(dataset, task):
    """Return the local path to the HDF5 file for a dataset/task."""
    cfg = DATASETS[dataset][task]
    return os.path.join("data", cfg["_folder"], cfg["filename"])


def get_gt_fn(dataset, task):
    """Return the path to the ground-truth file (same file as input)."""
    return get_fn(dataset, task)


def prepare(dataset, task):
    """Download the dataset file if a remote URL is configured and the file
    does not already exist locally."""
    cfg = DATASETS[dataset][task]
    url = cfg.get("url")
    if url is not None:
        fn = get_fn(dataset, task)
        if not os.path.exists(fn):
            os.makedirs(Path(fn).parent, exist_ok=True)
            print(f"Downloading {url} -> {fn} ...")
            urlretrieve(url, fn)


def load_sparse_matrix(h5_group):
    """Reconstruct a SciPy CSR matrix from an HDF5 group."""
    indptr = h5_group["indptr"][:]
    indices = h5_group["indices"][:]
    data = h5_group["data"][:]
    shape = tuple(h5_group.attrs["shape"])
    return csr_matrix((data, indices, indptr), shape=shape)


def get_h5_item(f, path):
    """Traverse a slash-separated or list path through an HDF5 file."""
    if isinstance(path, list):
        cur = f
        for p in path:
            cur = cur[p]
        return cur
    cur = f
    for p in path.split("/"):
        cur = cur[p]
    return cur


def get_query_count(dataset, task):
    """Return the number of queries for a dataset/task by inspecting the HDF5 file."""
    cfg = DATASETS[dataset][task]
    fn = get_fn(dataset, task)
    with h5py.File(fn) as f:
        if "queries" in cfg:
            item = get_h5_item(f, cfg["queries"])
            # Dense dataset: has .shape directly
            if hasattr(item, "shape"):
                return item.shape[0]
            # Sparse group: shape stored in attrs
            return item.attrs["shape"][0]
        else:
            # task1: all-kNN, queries == corpus
            item = get_h5_item(f, cfg["data"])
            if hasattr(item, "shape"):
                return item.shape[0]
            return item.attrs["shape"][0]


def load_data(dataset, task):
    """Open the HDF5 file and return (corpus, queries, gt_I, cfg).

    For sparse tasks the corpus and queries are returned as SciPy CSR matrices;
    for dense tasks they are returned as raw h5py datasets (callers cast to
    numpy as needed).  The caller is responsible for closing the file.
    """
    cfg = DATASETS[dataset][task]
    fn = get_fn(dataset, task)
    f = h5py.File(fn)

    data = get_h5_item(f, cfg["data"])
    if cfg.get("sparse"):
        data = load_sparse_matrix(data)

    queries = None
    if "queries" in cfg:
        queries = get_h5_item(f, cfg["queries"])
        if cfg.get("sparse"):
            queries = load_sparse_matrix(queries)

    gt_I = get_h5_item(f, cfg["gt_I"])

    return f, data, queries, gt_I, cfg
