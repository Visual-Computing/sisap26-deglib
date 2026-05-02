"""
Performance benchmark for EvpBits similarity computation.

Compares two approaches:
1. Multi-threaded nested loop using evp_similarity().
2. Batched matrix multiplication (ternary dot product).

This script demonstrates why the matrix approach is preferred for large-scale 
similarity computations.

Usage:
    uv run python tests/test_speed.py
"""
import time
import numpy as np
import h5py
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing the evp package
sys.path.append(str(Path(__file__).parent.parent.parent))

from evp import get_h5_file, EvpBits, evp_similarity, evp_similarity_batch

NON_ZEROS = 512
CHUNK_SIZE = 8192

def test_speed():
    BATCH_SIZE = 100
    
    print(f"Loading first {CHUNK_SIZE} EvpBits objects from the H5 dataset...")
    source_path = get_h5_file(is_small=True)
    with h5py.File(source_path, 'r') as f:
        # Only load the first CHUNK_SIZE elements to keep the test fast
        dataset = f['train'][:CHUNK_SIZE]
        N, dim = dataset.shape
        print(f"Loaded {N} vectors (dim={dim}). Converting to EvpBits...")
        evp_list = EvpBits.from_embeddings(dataset, NON_ZEROS, chunk_size=CHUNK_SIZE)
    
    # Take the first BATCH_SIZE items to compare against all N items
    queries = evp_list[:BATCH_SIZE]
    
    print(f"\n--- Method 1: Nested Loop with evp_similarity (Single-threaded) ---")
    start_time = time.time()
    results_loop = np.zeros((BATCH_SIZE, N), dtype=np.float32)    
    for i in range(BATCH_SIZE):
        q = queries[i]
        for j in range(N):
            results_loop[i, j] = evp_similarity(q, evp_list[j])            
    loop_time = time.time() - start_time
    print(f"Time taken: {loop_time:.4f} seconds")
    
    
    print(f"\n--- Method 2: Matrix Multiplication (evp_similarity_batch) ---")
    start_time = time.time()
    results_matrix = evp_similarity_batch(queries, evp_list)    
    matrix_time = time.time() - start_time
    print(f"Time taken (including matrix setup): {matrix_time:.4f} seconds")
    

    print(f"\n--- Comparison ---")
    print(f"Matrix approach is {loop_time / matrix_time:.2f}x faster!")
    
    # Check correctness
    diff = np.abs(results_loop - results_matrix).max()
    print(f"Max difference between results: {diff}")
    if diff == 0:
        print("Mathematical equivalence: CONFIRMED")
    else:
        print("Mathematical equivalence: FAILED")

if __name__ == '__main__':
    test_speed()
