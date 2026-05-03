# SISAP 2026: Wikipedia BGE-M3 Similarity Search

This repository contains implementations for the SISAP 2026 Challenge, focusing on approximate nearest neighbor search (ANN) on the Wikipedia BGE-M3 dataset. It features two primary approaches: **deglib** (Dynamic Exploration Graph Library) and **EVP** (Equi-Voronoi Polytope) Quantisation.

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

The following results were obtained on the **Wikipedia BGE-M3 Small** dataset (200,000 elements) using an **AMD Ryzen 5 5600G** with **32GB RAM**.

| Method | Settings | Build Time | Query Time | Throughput | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear (Rust)** | `fp32` | 1.1 s | 725.1 s | ? QPS | 1.0 |
| **EVP** | `NON_ZEROS=512` | 5.9 s | 340.1 s | 588 QPS | 0.7271 |
| **EVP Rust** | `NON_ZEROS=512` | 24.1 s | 175.1 s | ? QPS | 0.7270 |
| **deglib Explore** | `M=32`, `MaxDist=100` | 38.6 s | 18.2 s | 10,992 QPS | 0.7808 |
| **deglib Neighbors** | `M=48` | 74.6 s | 5.4 s | 36,704 QPS | 0.7861 |

*Note: Build time includes data loading and conversion/graph construction. Query time for EVP is for all-pairs similarity, while for deglib it is for retrieving K=15 neighbors for all elements.*

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
