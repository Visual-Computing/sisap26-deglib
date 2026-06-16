# ============================================================
# SISAP 2026 deglib submission — single container.
# Stage 1 builds the deglib_sisap C++ binary (default -march=native; pass
# --build-arg FORCE_AVX2=ON for an AVX2-only build). Stage 2 is a thin Python
# runtime that runs submission/search.py, which drives the binary. Both stages
# share the same Ubuntu base so the binary's glibc/libstdc++ match the runtime.
# Build from the repo root:  docker build -t sisap-deglib .
# ============================================================
FROM ubuntu:24.04@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git cmake g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY cpp /build/cpp
WORKDIR /build/cpp
# ISA selection:
#   FORCE_AVX2=ON (default) -> AVX2-only build (drops -march=native so AVX-512 is
#     not re-enabled). Default because we build the submission image on an AVX-512
#     host (our build box) but it must run on the EPYC 7F72 eval node, which has NO
#     AVX-512 — a -march=native binary would crash there with SIGILL.
#   FORCE_AVX2=OFF -> -march=native: best ISA on the build host (only safe when the
#     build host and the run host share the same CPU).
# Usage (AVX2-only):  docker build --build-arg FORCE_AVX2=ON -t sisap-deglib .
ARG FORCE_AVX2=OFF
RUN if [ "$FORCE_AVX2" = "ON" ]; then \
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFORCE_AVX2=ON ; \
    else \
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" ; \
    fi \
    && cmake --build build --target deglib_sisap -j"$(nproc)"

# ============================================================
# Stage 2: runtime — Python entrypoint + binary + harness
# ============================================================
FROM ubuntu:24.04@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# python3 + numpy + h5py for search.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
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
# Unbuffered stdout so search.py's progress lines interleave in order with the
# C++ binary's output in the TIRA logs (otherwise Python's block buffering flushes
# them after the subprocess, making one run look like two).
ENV PYTHONUNBUFFERED=1

# TIRA invokes the container as:
#   python3 /app/search.py --input $inputDataset/*.h5 \
#       --task-description $inputDataset/config.json --output $outputDir
# (no fixed ENTRYPOINT; the command is supplied by the evaluation harness)
