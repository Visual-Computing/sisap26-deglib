# Task 2 Submission Summary — Llama Dev dataset

This table lists the best sweep configuration for each submission test that minimizes search time while reaching at least 80% recall.

**Host:** INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (4C/8T) &nbsp;·&nbsp; RAM: 29.4 GiB total
**Container limits:** 8 CPU threads · 24.0 GiB RAM

| Mode | Method | Best Settings | Load Time | Build Time | FLAS Time | Total Time | Search Time | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mode5 | Mode 5: L2 Build (d+1) & FP16 IP Explore (+ FLAS) | eps=0.18, max_dist=7000 | 0.1s | 19.0s | 38.3s | 19.3s | 20.50 ms | 80.33% |
| mode7 | Mode 7: L2 Build (d+2) & FP16 L2 Explore (+ FLAS) | eps=0.007, max_dist=6500 | 0.1s | 19.0s | 38.1s | 19.4s | 22.90 ms | 80.15% |