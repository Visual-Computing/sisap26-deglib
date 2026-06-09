# Task 2 Benchmark Summary — Llama Dev dataset

This table lists the best sweep configuration for each test that minimizes search time while reaching at least 80% recall.

**Host:** INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (4C/8T) &nbsp;·&nbsp; RAM: 29.4 GiB total
**Container limits:** 8 CPU threads · 24.0 GiB RAM

| Mode | Method | Best Settings | Load Time | Build Time | FLAS Time | Total Time | Search Time | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mode3 | Mode 3: FP32 Build & FP16 Explore (no FLAS) | eps=0.25, max_dist=25000 | 0.1s | 19.7s | — | 19.9s | 60.10 ms | 80.10% |
| mode3 | Mode 3: FP32 Build & FP16 Explore (+ FLAS) | eps=0.28, max_dist=18000 | 0.1s | 21.8s | 38.8s | 22.0s | 53.80 ms | 81.10% |
| mode5 | Mode 5: L2 Build (d+1) & FP16 IP Explore (no FLAS) | eps=0.18, max_dist=8000 | 0.1s | 22.7s | — | 22.9s | 23.50 ms | 80.12% |
| mode5 | Mode 5: L2 Build (d+1) & FP16 IP Explore (+ FLAS) | eps=0.18, max_dist=7000 | 0.1s | 18.7s | 38.7s | 19.0s | 20.50 ms | 80.33% |
| mode4 | Mode 4: L2 Build (d+1) & FP32 L2 Explore (+ FLAS) | eps=0.008, max_dist=6500 | 0.1s | 19.3s | 39.2s | 19.5s | 27.50 ms | 80.88% |
| mode6 | Mode 6: L2 Build (d+1) & FP16 L2 Explore (+ FLAS) | eps=0.007, max_dist=8000 | 0.1s | 18.7s | 38.8s | 19.0s | 20.90 ms | 80.32% |
| mode7 | Mode 7: L2 Build (d+2) & FP16 L2 Explore (+ FLAS) | eps=0.007, max_dist=6500 | 0.1s | 19.0s | 39.0s | 19.3s | 22.50 ms | 80.15% |