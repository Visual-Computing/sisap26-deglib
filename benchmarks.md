# Deglib Benchmark Results

This document contains performance benchmarks for different configurations, implementations, and hardware environments.

## Table of Contents

- [AMD Ryzen 5 5600G](#amd-ryzen-5-5600g)
  - [Python Benchmarks](#python-benchmarks)
- [AMD Ryzen AI 9 HX Pro 375](#amd-ryzen-ai-9-hx-pro-375)
  - [C++ Core Benchmarks](#c-core-benchmarks)
- [Intel Xeon Platinum 8581C](#intel-xeon-platinum-8581c)
  - [Python Benchmarks](#python-benchmarks-1)
  - [Small Dataset (200K vectors)](#small-dataset-200k-vectors)
  - [Large Dataset (6.4M vectors)](#large-dataset-64m-vectors)

---

## AMD Ryzen 5 5600G

- **CPU:** AMD Ryzen 5 5600G with AVX2 instruction support
- **RAM:** 32 GB

### Python Benchmarks

| Method | Settings | DType | Build Time | Query Time | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EVP (Python)** | `NON_ZEROS=512` | `evp` |  5.9 s | 340.1 s | 0.7084 |
| **deglib Explore (Python)** | `M=32`, `MaxDist=100` | `fp32` | 38.6 s | 18.2 s | 0.7808 |
| **deglib Neighbors (Python)** | `M=48` | `fp32` | 74.6 s | 5.4 s | 0.7861 |

---

## AMD Ryzen AI 9 HX Pro 375

- **CPU:** AMD Ryzen AI 9 HX Pro 375 with AVX512 instruction support
- **RAM:** 64 GB

### C++ Core Benchmarks

| Method | Settings | Quant Time | Build Time | Convert Time | Explore Time | Rerank Time | **Total Time** | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **deglib FP32 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | 0 s | 13.3 s | 0 s | 1.0 s | 0 s | **14.3 s** | 0.8290 |
| **deglib FP16 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | 0 s | 8.2 s | 0 s | 0.7 s | 0 s | **8.9 s** | 0.8290 |
| **evp linear search (cpp)** | — | 0.8 s | 0.0 s | 0 s | 108.0 s | 0 s | **109.0 s** | 0.7084 |
| **deglib+evp Build&Explore (cpp)** | `M=32`, `MaxDist=200` | 0.8 s | 4.6 s | 0 s | 0.7 s | 0 s | **6.1 s** | 0.6702 |
| **deglib+evp Build&Explore+FP32 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 0.8 s | 4.6 s | 0 s | 1.2 s | 2.7 s | **9.3 s** | 0.8209 |
| **deglib+evp Build&Explore+FP16 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 0.8 s | 4.6 s | 0 s | 1.2 s | 2.3 s | **8.9 s** | 0.8206 |
| **deglib+evp build+FP32 Explore (cpp)** | `M=32`, `MaxDist=200` | 0.8 s | 4.6 s | 0.1 s | 2.5 s | 0 s | **8.0 s** | 0.8255 |
| **deglib+evp build+FP16 Explore (cpp)** | `M=32`, `MaxDist=200` | 0.8 s | 4.6 s | 0.1 s | 1.7 s | 0 s | **7.2 s** | 0.8255 |

---

## Intel Xeon Platinum 8581C

- **CPU:** Intel Xeon Platinum 8581C (allocated 8 of 60 cores) with AVX512 instruction support
- **RAM:** 30 GB

### Python Benchmarks

| Method | Settings | DType | Build Time | Query Time | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EVP (Python)** | `NON_ZEROS=512` | `evp` |  4.5 s | 292.7 s | 0.7271 |
| **deglib Explore (Python)** | `M=32`, `MaxDist=100` | `fp32` | 15.0 s | 9.3 s | 0.7808 |
| **deglib Neighbors (Python)** | `M=48` | `fp32` | 28.4 s | 2.4 s | 0.7861 |

### Small Dataset (200K vectors)
- **Benchmark Script:** `benchmark_task1_small.py` (configured to use 8 cores and 24 GB RAM)

| Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Total | Recall |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | FP16 Build+Explore | `M=32`, `MaxDist=100` | 0.2 s | 0.0 s | 9.2 s | 0.0 s | 0.7 s | 0.0 s | **10.1 s** | 85.31% |
| **2** | EVP linear search | — | 0.2 s | 0.5 s | 0.0 s | 0.0 s | 147.6 s | 0.0 s | **148.3 s** | 71.24% |
| **3** | EVP Build+Explore | `M=32`, `MaxDist=200` | 0.2 s | 0.6 s | 3.7 s | 0.0 s | 0.3 s | 0.0 s | **4.7 s** | 68.17% |
| **4** | EVP Build+Explore+Rerank | `M=32`, `MaxDist=200`, `evpK=50` | 0.2 s | 0.6 s | 3.8 s | 0.0 s | 0.5 s | 0.4 s | **5.6 s** | 84.15% |
| **5** | EVP build+FP16 Explore | `M=32`, `MaxDist=200` | 0.2 s | 0.6 s | 3.9 s | 0.1 s | 1.4 s | 0.0 s | **6.2 s** | 84.86% |
| **6** | EVP build+Asym Explore | `M=32`, `MaxDist=200` | 0.2 s | 0.6 s | 3.7 s | 0.0 s | 1.0 s | 0.0 s | **5.5 s** | 73.75% |
| **7** | EVP build+Asym+Rerank | `M=32`, `MaxDist=200`, `evpK=50` | 0.2 s | 0.6 s | 3.7 s | 0.0 s | 1.2 s | 0.4 s | **6.3 s** | 84.85% |

### Large Dataset (6.4M vectors)
- **Benchmark Script:** `benchmark_task1_large.py` (configured to use 8 cores and 24 GB RAM)

| Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Total | Recall |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | FP16 Build+Explore | `M=32, MaxDist=100` | — | — | — | — | — | — | — | — |
| **3** | EVP Build+Explore | `M=32, MaxDist=200` | — | — | — | — | — | — | — | — |
| **4** | EVP Build+Explore+Rerank | `M=32, MaxDist=200, evpK=50` | — | — | — | — | — | — | — | — |
| **5** | EVP build+FP16 Explore | `M=32, MaxDist=200` | — | — | — | — | — | — | — | — |
| **6** | EVP build+Asym Explore | `M=32, MaxDist=200` | — | — | — | — | — | — | — | — |
| **7** | EVP build+Asym+Rerank | `M=32, MaxDist=200, evpK=50` | — | — | — | — | — | — | — | — |