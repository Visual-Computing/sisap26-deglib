# Task 1 Small Benchmark Summary — Small Dataset (200K vectors)

This table lists the metrics for each benchmark mode.

**Host:** INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (4C/8T) &nbsp;·&nbsp; RAM: 29.4 GiB total
**Container limits:** 8 CPU threads · 24.0 GiB RAM

| Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Total | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | FP16 Build+Explore | M=32, MaxDist=100 | 0.2s | 0.0s | 8.6s | 0.0s | 0.7s | 0.0s | 9.5s | 0.85% |
| 3 | EVP Build+Explore | M=32, MaxDist=200 | 0.2s | 0.5s | 3.8s | 0.0s | 0.4s | 0.0s | 4.9s | 0.68% |
| 4 | EVP Build+Explore+Rerank | M=32, MaxDist=200, evpK=50 | 0.2s | 0.5s | 3.6s | 0.0s | 0.6s | 0.4s | 5.3s | 0.84% |
| 5 | EVP build+FP16 Explore | M=32, MaxDist=200 | 0.2s | 0.5s | 3.6s | 0.1s | 1.4s | 0.0s | 5.8s | 0.85% |
| 6 | EVP build+Asym Explore | M=32, MaxDist=200 | 0.2s | 0.5s | 3.7s | 0.0s | 1.0s | 0.0s | 5.5s | 0.74% |
| 7 | EVP build+Asym+Rerank | M=32, MaxDist=200, evpK=50 | 0.2s | 0.5s | 3.9s | 0.0s | 1.1s | 0.4s | 6.3s | 0.85% |