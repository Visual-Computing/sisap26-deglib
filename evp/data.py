from huggingface_hub import hf_hub_download

def get_h5_file(is_small: bool = True):
    """
    Downloads or locates the SISAP 2026 dataset from Hugging Face Hub.
    Returns the absolute path to the downloaded/cached HDF5 file.
    """
    filename = "benchmark-dev-wikipedia-bge-m3-small.h5" if is_small else "benchmark-dev-wikipedia-bge-m3.h5"
    
    source_path = hf_hub_download(
        repo_id="SISAP-Challenges/SISAP2026",
        filename=filename,
        repo_type="dataset",
    )
    return source_path
