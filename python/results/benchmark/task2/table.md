### Übersicht Zeiten / Recall (Task 2)

| Mode | Method | Best Settings | Load Time | Build Time | FLAS Time | Total Time | Search Time | Recall |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| mode3 | Mode 3: FP32 IP Build & FP16 IP Explore (no FLAS) | eps=0.25, max_dist=25000 | 0.06s | 19.50s | — | 19.80s | 99.60 ms | 80.10% |
| mode3 | Mode 3: FP32 IP Build & FP16 IP Explore (+ L2 FLAS) | eps=0.28, max_dist=18000 | 0.06s | 22.40s | 38.90s | 22.60s | 84.90 ms | 81.10% |
| mode10 | Mode 10: FP32 IP Build (d+1) & FP16 IP Explore (no FLAS) | eps=0.18, max_dist=8000 | 0.06s | 25.60s | — | 25.90s | 35.70 ms | 81.50% |
| mode10 | Mode 10: FP32 IP Build (d+1) & FP16 IP Explore (+ L2 FLAS) | eps=0.18, max_dist=6000 | 0.06s | 20.50s | 38.10s | 20.80s | 29.00 ms | 80.08% |
| mode10 | Mode 10: FP32 IP Build (d+1) & FP16 IP Explore (+ IP FLAS) | eps=0.18, max_dist=7000 | 0.06s | 20.40s | 38.40s | 20.80s | 32.30 ms | 81.13% |
| mode5 | Mode 5: FP32 L2 Build (d+1) & FP16 IP Explore (+ L2 FLAS) | eps=0.18, max_dist=7000 | 0.06s | 18.60s | 38.00s | 18.90s | 32.30 ms | 80.33% |
| mode6 | Mode 6: FP32 L2 Build (d+1) & FP16 L2 Explore (+ L2 FLAS) | eps=0.007, max_dist=8000 | 0.06s | 19.20s | 39.00s | 19.60s | 32.10 ms | 80.32% |
