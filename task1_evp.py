import time
import os
import h5py
import numpy as np
import psutil
import cpuinfo
from evp import EvpBits, compute_all_similarities_batch
from utils.data import get_h5_file

CHUNK_SIZE = 1024
NON_ZEROS = 512
K_TOP = 15
K_RECALL = 15
SMALL_DATASET = True

# Output:
# =============================================
#          FINAL EVALUATION SUMMARY           
# =============================================
# Loading & Conversion:              5.66 s
# Similarity Computation:          348.61 s
# Recall Calculation:                1.01 s
# ---------------------------------------------
# AVERAGE RECALL@15:                 0.7270
# =============================================

def main():
    print("Downloading/Locating Wikipedia data from Hugging Face Hub...", flush=True)
    source_path = get_h5_file(is_small=SMALL_DATASET)
    
    # 1. Loading & Conversion
    print("Loading Wikipedia data and building EvpBits...", flush=True)
    load_start = time.time()    
    with h5py.File(source_path, 'r') as f:
        # Note: from_embeddings reads from H5 in chunks, so this combines loading + conversion
        data_evp = EvpBits.from_embeddings(f['train'], NON_ZEROS, chunk_size=CHUNK_SIZE)        
        gt_knns = f['allknn/knns'][:]            
    load_and_convert_time = time.time() - load_start
    
    # 2. Similarity Computation
    print(f"Computing all-pairs similarities (N={len(data_evp)}) to find Top {K_TOP}...", flush=True)
    sim_start = time.time()
    top_100_indices = compute_all_similarities_batch(data_evp, k_top=K_TOP, batch_size=CHUNK_SIZE)    
    sim_time = time.time() - sim_start
    
    # 3. Recall Calculation
    print("Computing Recall...", flush=True)
    recall_start = time.time()    
    num_rows = len(data_evp)
    total_recall = 0.0
    for i in range(num_rows):
        # SISAP ground truth is 1-indexed, so we subtract 1
        gt_top = set(gt_knns[i, :K_RECALL] - 1)
        my_top = set(top_100_indices[i, :K_RECALL])
        
        overlap = len(gt_top.intersection(my_top))
        total_recall += overlap / K_RECALL
    avg_recall = total_recall / num_rows
    recall_calc_time = time.time() - recall_start

    # Final Summary Output
    cpu_info = cpuinfo.get_cpu_info()['brand_raw']
    ram_gb = psutil.virtual_memory().total / (1024**3)

    print("\n" + "="*45)
    print(f"{'SUMMARY':^45}")
    print("="*45)
    print(f"{'Dataset:':<25} {'Small' if SMALL_DATASET else 'Large'} ({num_rows:,} elements)")
    print(f"{'Hardware:':<25} {cpu_info}")
    print(f"{'RAM:':<25} {ram_gb:.2f} GB")
    print("-" * 45)
    print(f"{'Settings:':<25} NON_ZEROS={NON_ZEROS}, K_TOP={K_TOP}")
    print("-" * 45)
    print(f"{'Loading & Conversion:':<25} {load_and_convert_time:>10.2f} s")
    print(f"{'Similarity Computation:':<25} {sim_time:>10.2f} s")
    print(f"{'Recall Calculation:':<25} {recall_calc_time:>10.2f} s")
    print("-" * 45)
    print(f"{f'AVERAGE RECALL@{K_RECALL}:':<25} {avg_recall:>10.4f}")
    print("="*45)

if __name__ == '__main__':
    main()
