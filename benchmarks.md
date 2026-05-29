This file contains old benchmark numbers

**AMD Ryzen 5 5600G** with AVX2 instructions and **32GB RAM**.

| Method | Settings | DType | Build Time | Query Time | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EVP (Python)** | `NON_ZEROS=512` | `evp` |  5.9 s | 340.1 s | 0.7084 |
| **deglib Explore (Python)** | `M=32`, `MaxDist=100` | `fp32` | 38.6 s | 18.2 s | 0.7808 |
| **deglib Neighbors (Python)** | `M=48` | `fp32` | 74.6 s | 5.4 s | 0.7861 |

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
