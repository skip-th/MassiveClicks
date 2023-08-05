/** Parallelizing EM on GPU(s).
 *
 * parallel_em.cuh:
 *  - Defines the functions used for initiating the EM process on the GPU.
 */

#ifndef CLICK_MODEL_PARALLEL_EM_H
#define CLICK_MODEL_PARALLEL_EM_H

// System include.
#include <iostream>
#include <string>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <cuda_runtime.h>

// User include.
#include "../utils/definitions.h"
#include "../utils/macros.cuh"
#include "../utils/types.h"
#include "../utils/utils.cuh"
#include "../utils/timer.h"
#include "../click_models/base.cuh"
#include "../click_models/param.cuh"
#include "../click_models/evaluation.h"
#include "../data/dataset.h"
#include "../data/search.cuh"
#include "communicator.h"
#include "kernel.cuh"

/**
 * @brief Execute EM algorithm in parallel.
 *
 * @param config: Processing configuration.
 * @param device_partitions: The datasets assigned to the node's devices.
 */
 void em_parallel(
    const ProcessingConfig& config,
    LocalPartitions& device_partitions
);

#endif // CLICK_MODEL_PARALLEL_EM_H