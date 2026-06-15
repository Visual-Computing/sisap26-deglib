# ============================================================
# SISAP 2026 deglib submission — single container.
# Stage 1 builds the deglib_sisap C++ binary (AVX2, no AVX-512 — the EPYC 7F72
# eval node has no AVX-512). Stage 2 is a thin Python runtime that runs
# submission/search.py, which drives the binary. Both stages share the same
# Ubuntu base so the binary's glibc/libstdc++ match the runtime.
# Build from the repo root:  docker build -t sisap-deglib .
# ============================================================
FROM ubuntu:24.04@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git cmake g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY cpp /build/cpp
WORKDIR /build/cpp
# AVX2-only: drop -march=native so AVX-512 is not re-enabled on AVX-512 build hosts.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFORCE_AVX2=ON \
    && cmake --build build --target deglib_sisap -j"$(nproc)"

# ============================================================
# Stage 2: runtime — Python entrypoint + binary + harness
# ============================================================
FROM ubuntu:24.04@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# libgomp1 = OpenMP runtime for the binary; python3 + numpy + h5py for search.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        python3 \
        python3-numpy \
        python3-h5py \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/cpp/build/bin/deglib_sisap /usr/local/bin/deglib_sisap

WORKDIR /app
COPY submission/ /app/
RUN chmod +x /app/search.py

# search.py finds the binary here by default.
ENV DEGLIB_BIN=/usr/local/bin/deglib_sisap

# TIRA invokes the container as:
#   python3 /app/search.py --input $inputDataset/*.h5 \
#       --task-description $inputDataset/config.json --output $outputDir
# (no fixed ENTRYPOINT; the command is supplied by the evaluation harness)
