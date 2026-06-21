# Task 1 Small Benchmark Summary — Small Dataset (200K vectors)

This table lists the metrics for each benchmark mode.

**Host:** INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (4C/8T) &nbsp;·&nbsp; RAM: 29.4 GiB total
**Container limits:** 8 CPU threads · 24.0 GiB RAM

| Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Total | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | FP16 Build+Explore | M=32, MaxDist=92 | 0.2s | 0.0s | 8.5s | 0.0s | 0.6s | 0.0s | 9.3s | 84.87% |
| 2 | EVP linear search | — | 0.2s | 0.5s | 0.0s | 0.0s | 175.4s | 0.0s | 176.2s | 71.24% |
| 3 | EVP Build+Explore | M=32, MaxDist=200 | 0.2s | 0.5s | 3.7s | 0.0s | 0.4s | 0.0s | 4.9s | 68.16% |
| 4 | EVP Build+Explore+Rerank | M=32, MaxDist=220, evpK=50 | 0.2s | 0.6s | 3.9s | 0.0s | 0.6s | 0.4s | 5.7s | 84.84% |
| 5 | EVP build+FP16 Explore | M=32, MaxDist=200 | 0.2s | 0.5s | 3.8s | 0.1s | 1.4s | 0.0s | 6.0s | 84.87% |
| 6 | EVP build+Asym Explore | M=32, MaxDist=200 | 0.2s | 0.5s | 3.9s | 0.0s | 1.0s | 0.0s | 5.7s | 73.78% |
| 7 | EVP build+Asym+Rerank | M=32, MaxDist=200, evpK=50 | 0.2s | 0.6s | 3.8s | 0.0s | 1.1s | 0.4s | 6.1s | 84.85% |
| 8 | EVP asymmetric linear search | — | 0.2s | 0.5s | 0.0s | 0.0s | 857.4s | 0.0s | 858.1s | 78.54% |