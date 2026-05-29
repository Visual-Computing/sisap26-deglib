# ============================================================
# Stage 1: builder — compile deglib_evp_task1
# ============================================================
FROM ubuntu:24.04 AS builder

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        cmake \
        g++ \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone the evp branch with all submodules (fmt lives in external/fmt)
RUN git clone --recurse-submodules -b evp \
        https://github.com/Visual-Computing/DynamicExplorationGraph.git

WORKDIR /build/DynamicExplorationGraph/cpp

# Configure:
#   - Release build for maximum performance
#   - CMAKE_CXX_FLAGS="-march=native" → compile with the best instruction set (AVX2, AVX-512, SSE, etc.) supported by the host machine's CPU
#   - ENABLE_BENCHMARKS is left at its default (ON) so the evp/ subdirectory
#     (which contains deglib_evp_task1) is included in the build
#   - DATA_PATH is intentionally omitted — the HDF5 path is passed at runtime
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-march=native" \
    && cmake --build build --target deglib_evp_task1 -j$(nproc)

# ============================================================
# Stage 2: runtime — minimal image with only the binary
# ============================================================
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# libgomp1 is the OpenMP runtime required at execution time.
# It is NOT included in the bare ubuntu:24.04 image, so we install it
# explicitly to avoid "libgomp.so.1: cannot open shared object file" errors.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the compiled binary from the builder stage
COPY --from=builder \
    /build/DynamicExplorationGraph/cpp/build/bin/deglib_evp_task1 \
    /usr/local/bin/deglib_evp_task1

# Create the expected mount-points:
#   /data    → dataset directory (mounted read-only by the caller)
#   /results → output directory  (mounted read-write by the caller)
RUN mkdir -p /data /results

# Usage:
#   docker run --rm \
#       --cpus=8 --memory=24g --memory-swap=24g --memory-swappiness=0 \
#       --volume "<hf_cache_dir>:/data:ro" \
#       --volume "$(pwd)/results:/results:rw" \
#       sisap26-deglib \
#       /data/benchmark-dev-wikipedia-bge-m3-small.h5 mode4 --threads 8
ENTRYPOINT ["/usr/local/bin/deglib_evp_task1"]
