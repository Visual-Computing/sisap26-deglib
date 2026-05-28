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

**AMD Ryzen 5 5600G** with AVX2 instructions and **32GB RAM**.

**Wikipedia BGE-M3 Small** dataset (200,000 elements) in FP16 format
| Modi | Method | Settings | Load Time |Quant Time | Build Time | Convert Time | Explore Time | Rerank Time | **Overall Time** | Recall | Ideal RAM |
| :--- | :--- | :--- | :---: | :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **deglib FP16 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | 0.6 s | 0.0 s | 18.9 s | 0.1 s | 1.2 s | 0.0 s | **20.8 s** | 0.8295 | 460MB |
| 2 | **evp linear search (cpp)** | — | 0.6 s | 0.8 s | 0.0 s | 0.0 s | 209.3 s | 0.0 s | **211 s** | 0.7084 | 102MB |
| 3 | **deglib+evp Build&Explore (cpp)** | `M=32`, `MaxDist=200` | 0.6 s | 0.8 s | 4.8 s | 0.0 s | 0.9 s | 0.0 s | **7.1 s** | 0.6700 | 102MB |
| 4 | **deglib+evp Build&Explore+FP16 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 0.6 s | 0.8 s | 4.8 s | 0.0 s | 1.3 s | 3.7 s | **11.2 s** | 0.8209 | 512MB |
| 5 | **deglib+evp build+FP16 Explore (cpp)** | `M=32`, `MaxDist=200` | 0.6 s | 0.8 s | 4.8 s | 0.2 s | 3.8 s | 0.0 s | **10.2 s** | 0.8249 | 512MB |
| 6 | **deglib+evp build+Asym FP16&EVP Explore (cpp)** | `M=32`, `MaxDist=200` | 0.6 s | 0.8 s | 4.8 s | 0.0 s | 1.3 s | 0.0 s | **7.5 s** | 0.7249 | 512MB |
| 7 | **deglib+evp build+Asym FP16&EVP Explore+FP Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=50` | 0.6 s | 0.8 s | 4.8 s | 0.0 s | 1.3 s | 0.9 s | **8.4 s** | 0.825 | 512MB |

**Wikipedia BGE-M3 Large** dataset (6,400,000 elements) in FP16 format
| Modi | Method | Settings | Quant Time | Build Time | Convert Time | Explore+Rerank Time | **Overall Time** | Recall | 
| :--- | :--- | :--- | :---: | :---: |:---: | :---: | :---: | :---: | 
| 1 | **deglib FP16 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | — | — | — | — | — | — |
| 2 | **evp linear search (cpp)** | — | — | — | — | — | — | — |
| 3 | **deglib+evp Build&Explore (cpp)** | `M=32`, `MaxDist=200` | 22 s | 265 s | 0 s | 44 s | **331** | 0.6270 |
| 4 | **deglib+evp Build&Explore+FP16 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 22 s | 265 s | 0 s | 602 s | **889 s** | 0.7343 |
| 5 | **deglib+evp build+FP16 Explore (cpp)** | `M=32`, `MaxDist=200` | 22 s | 265 s | 4 s | 650 s | **941 s** | 0.7391 | 
| 6 | **deglib+evp build+Asym FP16&EVP Explore (cpp)** | `M=32`, `MaxDist=200` | 22 s | 265 s | 2 s | 57 s | **346** | 0.6695 |
| 7 | **deglib+evp build+Asym FP16&EVP Explore+FP Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=50` | 22 s | 265 s | 2 s | 85 s | **374 s** | 0.7382 |


| Modus | Methode | Quant Time | Build Time | Convert Time | Explore Time | Rerank Time | Total Time | Recall@15 |
|:---:|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | EVP build+EVP explore | 22.76 s | 278.13 s | — | 46.77 s | — | 347.67 s | 0.6278 |
| 4 | EVP build+EVP explore+FP16 rerank (evpK=200) | 23.27 s | 318.91 s | — | 617.22 s | 121.79 s | 1081.19 s | 0.7343 |
| 5 | EVP build+FP16 external search | 22.79 s | 266.86 s | 3.92 s | 647.31 s | — | 940.87 s | 0.7391 |
| 6 | EVP build+asym FP16&EVP explore | 22.72 s | 282.88 s | 2.15 s | 54.52 s | — | 362.26 s | 0.6695 |
| 7 | EVP build+asym explore+FP16 rerank (evpK=50) | 22.75 s | 284.85 s | 2.43 s | 58.71 s | 30.00 s | 398.74 s | 0.7382 |


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
