"""
Verification script for SISAP 2026 Similarity Approximation.

This script compares the ground-truth inner products of float16 embeddings 
against the EvpBits (sparsified ternary) similarity scores. It helps to 
verify that the bitwise representation correctly approximates the 
original cosine similarity / inner product.

Usage:
    uv run python tests/test_similarity.py
"""
import time
import h5py
import numpy as np
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing the evp package
sys.path.append(str(Path(__file__).parent.parent.parent))

from evp import EvpBits, evp_similarity, get_max_similarity
from utils.data import get_h5_file

CHUNK_SIZE = 8192
NON_ZEROS = 512

def main():
    print("Downloading/Locating Wikipedia data from Hugging Face Hub...")
    source_path = get_h5_file(is_small=True)
    
    program_start = time.time()
    
    print("Loading Wikipedia data...")
    load_start = time.time()
    
    f16_start = time.time()
    
    # Load f16 data
    with h5py.File(source_path, 'r') as f:
        # Load the whole train dataset as f16
        data_f16 = f['train'][:].astype(np.float16)
        
    print(f"f16 data loaded in {time.time() - f16_start:.3f} s")
    print(f"First row of Wikipedia data : {data_f16[0]}")
    
    f32_start = time.time()
    
    # In Python, we load as float32 and convert to EvpBits
    # Using the optimized from_embeddings which processes in chunks
    with h5py.File(source_path, 'r') as f:
        dataset = f['train']
        data_evp = EvpBits.from_embeddings(dataset, NON_ZEROS, chunk_size=CHUNK_SIZE)
        
    num_data = len(data_evp)
    print(f"{num_data} EvpBits Vectors loaded in {time.time() - f32_start:.3f} s")
    
    if num_data >= 10:
        print("\n=== Similarity Comparison (Top 10 Vectors) ===")
        print(f"{'Index':<6} | {'f16 IP':<10} | {'EVP Raw':<10} | {'EVP Norm':<10}")
        print("-" * 45)
        
        dim = data_f16.shape[1]
        max_sim = get_max_similarity(dim, NON_ZEROS)
        
        for i in range(10):
            # 1. Compute exact inner product
            a_f32 = data_f16[0].astype(np.float32)
            b_f32 = data_f16[i].astype(np.float32)
            ip = float(np.sum(a_f32 * b_f32))
            
            # 2. Compute EvpBits similarity
            sim = evp_similarity(data_evp[0], data_evp[i])
            norm_sim = sim / max_sim
            
            print(f"{i:<6} | {ip:<10.4f} | {sim:<10.1f} | {norm_sim:<10.4f}")
            
        print("-" * 45)
        print(f"Max Possible Similarity (Identity): {max_sim:.1f}")
        
    print(f"Total program execution time: {time.time() - program_start:.2f} s")

if __name__ == '__main__':
    main()
