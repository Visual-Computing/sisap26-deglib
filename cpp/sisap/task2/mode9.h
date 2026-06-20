#pragma once

/**
 * @file mode9.h
 * @brief Task 2 Mode 9: EVP Asymmetric Linear Search (evp-asymmetric-linear-search)
 *
 * Behavior:
 * 1. Loads FP32 training vectors from "train" as the database.
 * 2. Loads FP32 query vectors from "queries".
 * 3. Quantizes database vectors to EVP bits representation.
 * 4. Converts query vectors to FP16.
 * 5. Runs an exact brute-force linear search over all quantized database vectors for each query.
 * 6. Uses asymmetric distance comparison via FP16EvpAsymmetricSimilarity::compare.
 * 7. Tracks load, quantization, and search time separately.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #include <immintrin.h>
#endif

#include "../hdf5_reader.h"
#include "../sisap_common.h"
#include "builder.h"
#include "concurrent.h"
#include "distances.h"
#include "graph/sizebounded_graph.h"
#include "quantization/evp_quantize.h"
#include "repository.h"

namespace task2::mode_evp_asym_linear {

struct ExplorationTimings {
    double search_ms = 0.0;
    float recall = -1.0f;
};

using deglib::distances::floats_to_fp16;

static ExplorationTimings run_search(
    const std::vector<std::byte>& quantized_db,
    const std::vector<std::vector<uint16_t>>& queries_fp16,
    size_t db_count,
    size_t query_count,
    size_t dims,
    uint32_t k_top,
    uint8_t threads,
    bool compute_recall,
    int num_runs = 1,
    const std::vector<std::vector<int32_t>>& gt_data = {},
    const std::string& output_path = "",
    double build_time_s = 0.0)
{
    // Aim for ~8 chunks per thread so even small query sets spread across all threads
    const size_t chunk_size =
        std::clamp((query_count + static_cast<size_t>(threads) * 8 - 1) / (static_cast<size_t>(threads) * 8), size_t{1}, size_t{8196});
    const size_t num_chunks = (query_count + chunk_size - 1) / chunk_size;
    std::vector<std::vector<uint32_t>> results(query_count);
    std::vector<std::vector<float>> results_dists(query_count);
    std::vector<double> run_times;

    const size_t bytes_per_evp = dims / 4;
    const uint32_t dims_u32 = static_cast<uint32_t>(dims);

    for (int run = 0; run < num_runs; ++run) {
        std::fill(results.begin(), results.end(), std::vector<uint32_t>());
        std::fill(results_dists.begin(), results_dists.end(), std::vector<float>());

        double t_run_start = sisap_common::now_ms();
        deglib::concurrent::parallel_for(static_cast<size_t>(0), num_chunks, static_cast<uint32_t>(threads), 1,
            [&](size_t chunk_id, size_t) {
                size_t start = chunk_id * chunk_size;
                size_t end = std::min(start + chunk_size, query_count);
                size_t num_items = end - start;

                for (size_t i = 0; i < num_items; ++i) {
                    size_t q_idx = start + i;
                    const std::byte* query_ptr = reinterpret_cast<const std::byte*>(queries_fp16[q_idx].data());

                    std::vector<std::pair<float, uint32_t>> top;
                    top.reserve(k_top + 1);

                    for (size_t j = 0; j < db_count; ++j) {
                        const std::byte* cand_ptr = quantized_db.data() + j * bytes_per_evp;
                        float dist = deglib::distances::FP16EvpAsymmetricSimilarity::compare(query_ptr, cand_ptr, &dims_u32);

                        if (top.size() < k_top) {
                            top.push_back({dist, static_cast<uint32_t>(j)});
                            std::push_heap(top.begin(), top.end());
                        } else if (dist < top.front().first) {
                            std::pop_heap(top.begin(), top.end());
                            top.back() = {dist, static_cast<uint32_t>(j)};
                            std::push_heap(top.begin(), top.end());
                        }
                    }

                    std::sort_heap(top.begin(), top.end());

                    auto& res = results[q_idx];
                    auto& rdist = results_dists[q_idx];
                    res.reserve(top.size());
                    rdist.reserve(top.size());
                    for (const auto& pair : top) {
                        // SISAP submission expects 1-based indices (matching external labels)
                        res.push_back(pair.second + 1);
                        rdist.push_back(pair.first);
                    }
                }
            });

        run_times.push_back(sisap_common::now_ms() - t_run_start);
    }

    double avg_ms = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();

    float recall = -1.0f;
    if (compute_recall) {
        recall = sisap_common::compute_recall(gt_data, results, k_top);
    } else {
        // Pad every row to exactly k_top so the knns/dists matrix is rectangular.
        for (size_t i = 0; i < query_count; ++i) {
            results[i].resize(k_top, std::numeric_limits<uint32_t>::max());
            results_dists[i].resize(k_top, std::numeric_limits<float>::max());
        }
        sisap_common::write_knns_dists(output_path, results, results_dists, build_time_s, avg_ms / 1000.0);
    }

    return { avg_ms, recall };
}

static int run(
    const std::filesystem::path& data_path,
    uint32_t threads,
    uint32_t non_zeros,
    uint32_t k_top,
    int num_runs,
    const std::vector<uint32_t>& max_dist_list,
    bool compute_recall,
    float goal_recall,
    const std::string& output_path = "")
{
    (void)max_dist_list; // Linear search does not use max_dist sweep

    const std::string h5path = data_path.string();
    std::printf("\n");

    auto datasets = hdf5_reader::scan_datasets(h5path);
    auto& train_info = hdf5_reader::find_dataset(datasets, "train");

    const char* query_name = "test/queries";
    const char* knn_name = "test/knns";

    const hdf5_reader::DatasetInfo* query_info_ptr = nullptr;
    size_t query_count = 0;

    if (query_name) {
        query_info_ptr = &hdf5_reader::find_dataset(datasets, query_name);
        query_count = static_cast<size_t>(query_info_ptr->num_rows);
    }

    double t_load_start = sisap_common::now_ms();
    size_t dims = static_cast<size_t>(train_info.num_cols);
    size_t count = static_cast<size_t>(train_info.num_rows);

    std::printf("Dataset: train = %zu vectors, queries = %zu (%s), dims = %zu\n",
                count, query_count, query_name ? query_name : "none", dims);

    // --------------------------------------------------------------------------
    // Load ground truth
    // --------------------------------------------------------------------------
    std::vector<std::vector<int32_t>> gt_data;
    if (compute_recall) {
        if (knn_name) {
            gt_data = sisap_common::load_ground_truth_by_name(h5path, datasets, knn_name, k_top);

            if (gt_data.size() != query_count) {
                std::fprintf(stderr,
                    "Error: queries (%zu) and %s (%zu) must have the same number of rows\n",
                    query_count, knn_name, gt_data.size());
                return 1;
            }
        } else {
            std::fprintf(stderr, "No ground truth dataset found for recall computation.\n");
            return 1;
        }
    }
    double load_ms = sisap_common::now_ms() - t_load_start;

    std::printf("=== EVP Asymmetric Linear Search - Task 2 Mode 9 ===\n");

    // --------------------------------------------------------------------------
    // Load ALL FP32 training vectors once
    // --------------------------------------------------------------------------
    double t_load_fp32 = sisap_common::now_ms();
    std::vector<float> database_fp32 = hdf5_reader::read_flat_fp32(h5path, train_info);
    double load_fp32_ms = sisap_common::now_ms() - t_load_fp32;
    load_ms += load_fp32_ms;

    // --------------------------------------------------------------------------
    // Load query vectors and convert to FP16
    // --------------------------------------------------------------------------
    if (!query_info_ptr) {
        std::fprintf(stderr, "Error: No query dataset found in HDF5 file.\n");
        return 1;
    }
    double t_query_load = sisap_common::now_ms();
    auto queries_fp32 = hdf5_reader::read_matrix_fp32(h5path, *query_info_ptr);
    std::vector<std::vector<uint16_t>> queries_fp16;
    queries_fp16.reserve(queries_fp32.size());
    for (const auto& q : queries_fp32) {
        queries_fp16.push_back(floats_to_fp16(q));
    }
    double query_load_ms = sisap_common::now_ms() - t_query_load;
    load_ms += query_load_ms;
    std::printf("Loaded %zu queries and converted to FP16 (dims=%zu)\n", queries_fp16.size(), dims);

    // --------------------------------------------------------------------------
    // Quantize database vectors
    // --------------------------------------------------------------------------
    if (non_zeros >= dims) {
        std::printf("Warning: --non-zeros (%u) is >= dataset dimensions (%zu). Clamping to %zu.\n", non_zeros, dims, dims / 2);
        non_zeros = static_cast<uint32_t>(dims / 2);
    }

    std::printf("Quantizing database to EVP (non_zeros = %u)...\n", non_zeros);
    double t_quant = sisap_common::now_ms();
    auto quantized_db = deglib::quantization::quantize_batch(database_fp32.data(), count, static_cast<uint32_t>(dims), non_zeros, threads);
    double quantize_ms = sisap_common::now_ms() - t_quant;

    database_fp32.clear();
    database_fp32.shrink_to_fit();

    // --------------------------------------------------------------------------
    // Exploration
    // --------------------------------------------------------------------------
    std::printf("Starting asymmetric linear search: k_top=%u, threads=%u\n", k_top, threads);

    double build_time_s = (load_ms + quantize_ms) / 1000.0;
    if (!compute_recall && !output_path.empty()) {
        std::filesystem::create_directories(output_path);
    }

    std::string point_output;
    if (!compute_recall && !output_path.empty()) {
        point_output = output_path + "/op_md0.bin"; // Dummy operating point naming
    }

    auto timings = run_search(quantized_db, queries_fp16, count, query_count, dims, k_top,
                              static_cast<uint8_t>(threads), compute_recall, num_runs, gt_data, point_output, build_time_s);

    std::printf("Asymmetric Linear Search completed: recall %.2f %% and search time %.1f ms\n", timings.recall * 100.0f, timings.search_ms);

    std::printf("\n");
    double total_time_ms = load_ms + quantize_ms + timings.search_ms;

    sisap_common::print_summary(
        "EVP Asymmetric Linear Search", 8,
        load_ms, quantize_ms, 0.0, 0.0, 0.0,
        timings.search_ms, 0.0, total_time_ms,
        compute_recall, k_top, timings.recall,
        threads, 0, 0,
        0, 0, 0.0f, 0, count, dims, 0,
        deglib::builder::OptimizationTarget::LowLID, 0.0, 0.0
    );

    return 0;
}

} // namespace task2::mode_evp_asym_linear
