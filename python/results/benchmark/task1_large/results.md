# Task 1 Large Benchmark Summary — Large Dataset (6.4M vectors)

This table lists the metrics for each benchmark mode.

**Host:** INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (4C/8T) &nbsp;·&nbsp; RAM: 29.4 GiB total
**Container limits:** 8 CPU threads · 24.0 GiB RAM

| Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Total | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | FP16 Build+Explore | M=32, MaxDist=100 | 32.9s | 0.0s | 448.9s | 0.0s | 28.8s | 0.0s | 510.7s | 0.77% |
| 3 | EVP Build+Explore | M=32, MaxDist=200 | 11.1s | 17.0s | 176.2s | 0.0s | 24.4s | 0.0s | 228.7s | 0.64% |
| 4 | EVP Build+Explore+Rerank | M=32, MaxDist=200, evpK=50 | 6.8s | 15.7s | 189.8s | 0.0s | 37.1s | 13.4s | 263.0s | 0.76% |
| 5 | EVP build+FP16 Explore | M=32, MaxDist=200 | 35.7s | 15.8s | 187.4s | 2.3s | 57.3s | 0.0s | 298.6s | 0.77% |
| 6 | EVP build+Asym Explore | M=32, MaxDist=200 | 33.5s | 15.6s | 191.8s | 1.2s | 45.5s | 0.0s | 287.6s | 0.69% |
| 7 | EVP build+Asym+Rerank | M=32, MaxDist=200, evpK=50 | 42.4s | 15.6s | 187.7s | 1.2s | 51.1s | 13.8s | 312.1s | 0.77% |