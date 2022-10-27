/** Parallelizing EM on GPU(s).
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * parallel_em.cuh:
 *  - Defines the functions used for initiating the EM process on the GPU.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_PARALLEL_EM_H
#define CLICK_MODEL_PARALLEL_EM_H

// System include.
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <map>

// User include.
#include "../utils/definitions.h"
#include "../utils/macros.cuh"
#include "../utils/cuda_utils.cuh"
#include "../click_models/base.cuh"
#include "../click_models/param.cuh"
#include "../click_models/evaluation.h"
#include "../data/dataset.h"
#include "../data/search.cuh"
#include "communicator.h"
#include "kernel.cuh"

void em_parallel(const int model_type, const int node_id, const int n_nodes,
    const int n_threads,const int* n_devices_network, const int n_itr,
    std::vector<std::tuple<std::vector<SERP>, std::vector<SERP>, int>>& device_partitions,
    const std::vector<std::unordered_map<int, std::unordered_map<int, int>>*>& root_mapping);

#endif // CLICK_MODEL_PARALLEL_EM_H