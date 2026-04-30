from huggingface_hub import hf_hub_download

import h5py

DATASET_REPO = "sisap-challenges/SISAP2026"
DATASET_FILE = "benchmark-dev-wikipedia-bge-m3-small.h5"


def download_dataset():
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
    )
    print(f"Dataset downloaded to: {path}")
    return path


def inspect_dataset(path):
    with h5py.File(path, "r") as f:
        print("\nDataset contents:")
        print(f"  Top-level keys: {list(f.keys())}")
        for key in f.keys():
            obj: h5py.Group | h5py.Dataset = f[key]  # type: ignore[assignment]
            if isinstance(obj, h5py.Group):
                print(f"\n  Group: {key}")
                for subkey in obj.keys():
                    subobj: h5py.Group | h5py.Dataset = obj[subkey]  # type: ignore[assignment]
                    if isinstance(subobj, h5py.Dataset):
                        print(f"    Dataset: {subkey}  shape={subobj.shape}  dtype={subobj.dtype}")
                    else:
                        print(f"    Group: {subkey}  -> {list(subobj.keys())}")
            elif isinstance(obj, h5py.Dataset):
                print(f"\n  Dataset: {key}  shape={obj.shape}  dtype={obj.dtype}")


def main():
    path = download_dataset()
    inspect_dataset(path)


if __name__ == "__main__":
    main()
