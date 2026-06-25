#pragma once

#include "config.h"

#include <cstddef>

#if defined(_MSC_VER)
    #include <intrin.h>
    // _mm_prefetch accepts T0, T1, T2, and NTA. 
    // The GCC locality level (0-3) is mapped directly to MSVC hints here.
    #define DEGLIB_COMPILER_PREFETCH(addr, rw, locality) \
        ((locality) == 3 ? _mm_prefetch((const char*)(addr), _MM_HINT_T0) : \
         (locality) == 2 ? _mm_prefetch((const char*)(addr), _MM_HINT_T1) : \
         (locality) == 1 ? _mm_prefetch((const char*)(addr), _MM_HINT_T2) : \
                           _mm_prefetch((const char*)(addr), _MM_HINT_NTA))
#elif defined(__GNUC__) || defined(__clang__)
    #define DEGLIB_COMPILER_PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#else
    #define DEGLIB_COMPILER_PREFETCH(addr, rw, locality)
#endif

namespace deglib {
namespace memory {

/**
 * @brief Prefetch a specified number of bytes into the cache.
 * 
 * @tparam Locality Temporal locality level (0 to 3).
 *                  - 3: High temporal locality. Load into all cache levels (L1, L2, L3).
 *                  - 2: Moderate temporal locality. Load into L2 and L3 caches.
 *                  - 1: Low temporal locality. Load into L3 cache only.
 *                  - 0: No temporal locality. Non-temporal data (minimize cache pollution).
 * @param ptr Pointer to the memory address to start prefetching from.
 * @param bytes The number of bytes to prefetch.
 */
template <int Locality>
inline static void prefetch_bytes(const void* ptr, const size_t bytes) {
  if (bytes == 0) return;
  const char* p = reinterpret_cast<const char*>(ptr);
  const size_t cache_lines = (bytes - 1) >> 6;
  if (cache_lines > 16) {
    for (size_t i = 0; i <= cache_lines; ++i) {
      DEGLIB_COMPILER_PREFETCH(p + i * 64, 0, Locality);
    }
  } else {
    switch (cache_lines) {
      case 16: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 15: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 14: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 13: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 12: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 11: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 10: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 9:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 8:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 7:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 6:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 5:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 4:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 3:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 2:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 1:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 0:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); break;
    }
  }
}

/**
 * @brief Prefetch a specified number of cache lines (64 bytes each).
 * 
 * @tparam Locality Temporal locality level (0 to 3).
 *                  - 3: High temporal locality. Load into all cache levels (L1, L2, L3).
 *                  - 2: Moderate temporal locality. Load into L2 and L3 caches.
 *                  - 1: Low temporal locality. Load into L3 cache only.
 *                  - 0: No temporal locality. Non-temporal data (minimize cache pollution).
 * @param ptr Pointer to the memory address to start prefetching from.
 * @param cache_lines The number of 64-byte cache lines to prefetch.
 */
template <int Locality>
inline static void prefetch_cache_lines(const void* ptr, const size_t cache_lines) {
  if (cache_lines == 0) return;
  const char* p = reinterpret_cast<const char*>(ptr);
  if (cache_lines > 17) {
    for (size_t i = 0; i < cache_lines; ++i) {
      DEGLIB_COMPILER_PREFETCH(p + i * 64, 0, Locality);
    }
  } else {
    switch (cache_lines - 1) {
      case 16: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 15: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 14: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 13: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 12: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 11: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 10: DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 9:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 8:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 7:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 6:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 5:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 4:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 3:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 2:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 1:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); p += 64;
      case 0:  DEGLIB_COMPILER_PREFETCH(p, 0, Locality); break;
    }
  }
}

/** @brief Prefetch bytes to L1 cache (and L2/L3). */
inline static void prefetch_l1_bytes(const void* ptr, const size_t bytes) { prefetch_bytes<3>(ptr, bytes); }
/** @brief Prefetch bytes to L2 cache (and L3). */
inline static void prefetch_l2_bytes(const void* ptr, const size_t bytes) { prefetch_bytes<2>(ptr, bytes); }
/** @brief Prefetch bytes to L3 cache only. */
inline static void prefetch_l3_bytes(const void* ptr, const size_t bytes) { prefetch_bytes<1>(ptr, bytes); }
/** @brief Prefetch bytes using non-temporal access hint (NTA). */
inline static void prefetch_nta_bytes(const void* ptr, const size_t bytes) { prefetch_bytes<0>(ptr, bytes); }

/** @brief Prefetch cache lines to L1 cache (and L2/L3). */
inline static void prefetch_l1_cache_lines(const void* ptr, const size_t cache_lines) { prefetch_cache_lines<3>(ptr, cache_lines); }
/** @brief Prefetch cache lines to L2 cache (and L3). */
inline static void prefetch_l2_cache_lines(const void* ptr, const size_t cache_lines) { prefetch_cache_lines<2>(ptr, cache_lines); }
/** @brief Prefetch cache lines to L3 cache only. */
inline static void prefetch_l3_cache_lines(const void* ptr, const size_t cache_lines) { prefetch_cache_lines<1>(ptr, cache_lines); }
/** @brief Prefetch cache lines using non-temporal access hint (NTA). */
inline static void prefetch_nta_cache_lines(const void* ptr, const size_t cache_lines) { prefetch_cache_lines<0>(ptr, cache_lines); }

/** @brief Prefetch 128 bytes (2 cache lines) to L1 cache (and L2/L3). */
inline static void prefetch(const void* ptr) {
  const char* p = reinterpret_cast<const char*>(ptr);
  DEGLIB_COMPILER_PREFETCH(p + 0, 0, 3);
  DEGLIB_COMPILER_PREFETCH(p + 64, 0, 3);
}

/** @brief Prefetch bytes to L1 cache (and L2/L3). */
inline static void prefetch(const void* ptr, const size_t bytes) {
  prefetch_bytes<3>(ptr, bytes);
}

}  // namespace memory
}  // namespace deglib
