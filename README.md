# SISAP 2026: Wikipedia BGE-M3 Similarity Search

This repository contains implementations for the [SISAP 2026 Challenge](https://sisap-challenges.github.io/2026/index.html), focusing on approximate nearest neighbor search (ANN) on the Wikipedia BGE-M3 dataset. It features two primary approaches: **deglib** (Dynamic Exploration Graph Library) and [**EVP** (Equi-Voronoi Polytope) Quantisation](https://github.com/MetricSearch/metric_space_rust).


## 🏆 Task Description (Task 1: K-Nearest Neighbor Graph)

The goal of this task (metric self-join) is to compute an approximate $k$-nearest neighbor graph for $k=15$ using all objects in the dataset as queries (excluding self-references).

* **Dataset:** WIKIPEDIA (6.4 million vectors, 1024 dimensions, normalized, BGE-M3 model).
* **Distance Metric:** Dot product (inner product).
* **Target Operating Point:** Fastest execution time that achieves an average recall of **at least 0.8**.
* **Evaluation Criteria:** Total wall-clock time (including preprocessing, indexing, search, and postprocessing/re-ranking) and recall against a gold standard.
* **Hardware Constraints (Evaluation Container):** 8 vCPUs, 24 GB RAM, read-only mounted dataset. Max execution time: 8 hours.


## 🚀 Quick Start

### 1. Prerequisites
Ensure you have `uv` installed. If not, follow the [official installation guide](https://astral.sh/uv).

### 2. Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Visual-Computing/sisap26-deglib
cd sisap26-deglib
uv sync
```

## 🛠 Project Structure

### EVP (Equi-Voronoi Polytope) Quantisation.
The `evp` module implements a high-performance, bit-packed ternary approximation for inner product similarity.
- **`evp/`**: Core library containing the `EvpBits` implementation and similarity kernels.
- **`task1_evp.py`**: Main evaluation script for EVP, computing all-pairs similarities on the Wikipedia dataset.
- **`tests/evp/`**: Comprehensive test suite for EVP functionality, speed, and accuracy.

### deglib (Dynamic Exploration Graph)
Graph-based ANN search using the `deglib` library.
- **`task1_deglib_all.py`**: Benchmarks multiple search methods (search, neighbors, explore) on the deglib graph.
- **`task1_deglib_explore.py`**: Focused benchmark using the `explore` method.
- **`task1_deglib_neighbors.py`**: Focused benchmark using graph neighbors.

## 📊 Running Benchmarks

### EVP Benchmark
To run the EVP similarity search and recall evaluation:
```bash
uv run python task1_evp.py
```

### deglib Benchmarks
To run the comprehensive deglib benchmark:
```bash
uv run python task1_deglib_all.py
```

## 📈 Benchmark Results

The following results were obtained on the **Wikipedia BGE-M3 Small** dataset (200,000 elements) in FP16 format.

**AMD Ryzen 5 5600G** with AVX2 instructions and **32GB RAM**.

| Method | Settings | DType | Build Time | Query Time | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EVP (Python)** | `NON_ZEROS=512` | `evp` |  5.9 s | 340.1 s | 0.7084 |
| **deglib Explore (Python)** | `M=32`, `MaxDist=100` | `fp32` | 38.6 s | 18.2 s | 0.7808 |
| **deglib Neighbors (Python)** | `M=48` | `fp32` | 74.6 s | 5.4 s | 0.7861 |

| Method | Settings | Quant Time | Build Time | Convert Time | Explore Time | Rerank Time | **Total Time** | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **deglib FP16 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | 0.0 s | 18.9 s | 0.1 s | 1.2 s | 0.0 s | **20.2 s** | 0.8295 |
| **evp linear search (cpp)** | — | 0.8 s | 0.0 s | 0.0 s | 209.3 s | 0.0 s | **210.1 s** | 0.7084 |
| **deglib+evp Build&Explore (cpp)** | `M=32`, `MaxDist=200`` | 0.8 s | 4.8 s | 0.0 s | 0.9 s | 0.0 s | **6.5 s** | 0.6700 |
| **deglib+evp Build&Explore+FP16 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 0.8 s | 4.8 s | 0.1 s | 1.3 s | 3.9 s | **10.9 s** | 0.8209 |
| **deglib+evp build+FP16 Explore (cpp)** | `M=32`, `MaxDist=200` | 0.8 s | 4.8 s | 0.2 s | 3.8 s | 0.0 s | **9.6 s** | 0.8249 |
| **deglib+evp build+Asym FP16&EVP Explore (cpp)** | `M=32`, `MaxDist=200` | 0.8 s | 4.8 s | 0.2 s | 31.1 s | 0.0 s | **36.9 s** | 0.780 |

**AMD Ryzen AI 9 HX Pro 375** with AVX512 instruction and **64GB RAM**.

| Method | Settings |Quant Time | Build Time | Convert Time | Explore Time | Rerank Time | **Total Time** | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **deglib FP32 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | 0 s | 13.3 s | 0 s | 1.0 s | 0 s | **14.3 s** | 0.829 |
| **deglib FP16 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | 0 s | 8.2 s | 0 s | 0.7 s | 0 s | **8.9 s** | 0.829 |
| **evp linear search (cpp)** |  | 0.8 s | 0.0 s | 0 s | 108 s | 0 s | **109 s** | 0.7084 |
| **deglib+evp Build&Explore (cpp)** | `M=32`, `MaxDist=200` | 0.8 s | 4.6 s | 0 s | 0.7 s | 0 s | **6.1 s** | 0.6702 |
| **deglib+evp Build&Explore+FP32 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 0.8 s | 4.6 s | 0 s | 1.2 s | 2.7 s | **9.3 s** | 0.8209 |
| **deglib+evp Build&Explore+FP16 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 0.8 s | 4.6 s | 0 s | 1.2 s | 2.3 s | **8.9 s** | 0.8206 |
| **deglib+evp build+FP32 Explore (cpp)** | `M=32`, `MaxDist=200` | 0.8 s | 4.6 s | 0.1 s | 2.5 s | 0 s | **8.0 s** | 0.8255 |
| **deglib+evp build+FP16 Explore (cpp)** | `M=32`, `MaxDist=200` | 0.8 s | 4.6 s | 0.1 s | 1.7 s | 0 s | **7.2 s** | 0.8255 |


**Intel XEON PLATINUM 8581C** with 8 of 60 cores, AVX512 instructions and **30GB RAM**.

| Method | Settings | DType | Build Time | Query Time | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EVP (Python)** | `NON_ZEROS=512` | `evp` |  4.5 s | 292.7 s | 0.7271 |
| **deglib Explore (Python)** | `M=32`, `MaxDist=100` | `fp32` | 15.0 s | 9.3 s | 0.7808 |
| **deglib Neighbors (Python)** | `M=48` | `fp32` | 28.4 s | 2.4 s | 0.7861 |

*Note: Build time includes data loading, data conversion, and graph construction. Query time for EVP includes calculating all-pair similarities, while for deglib it measures retrieving K = 15 neighbors for all elements.*

## 🧪 Testing
Run the EVP test suite to verify implementation correctness and performance:
```bash
# Core logic and conversion
uv run python tests/evp/test_evp.py

# Similarity approximation accuracy
uv run python tests/evp/test_similarity.py

# Performance benchmarking
uv run python tests/evp/test_speed.py
```

## 📄 Dataset
The scripts automatically download the SISAP 2026 Wikipedia dataset from the [Hugging Face Hub](https://huggingface.co/datasets/SISAP-Challenges/SISAP2026). By default, the `small` version is used for development.
