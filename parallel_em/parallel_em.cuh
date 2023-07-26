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
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <cuda_runtime.h>

// User include.
#include "../utils/definitions.h"
#include "../utils/macros.cuh"
#include "../utils/utils.cuh"
#include "../utils/timer.h"
#include "../click_models/base.cuh"
#include "../click_models/param.cuh"
#include "../click_models/evaluation.h"
#include "../data/dataset.h"
#include "../data/search.cuh"
#include "communicator.h"
#include "kernel.cuh"

// ProcessingConfig encapsulates various parameters needed for executing the
// EM algorithm in parallel. This includes identifiers for the model and node,
// the number of nodes, threads, iterations, and devices, the execution mode,
// and an information of other devices in the cluster.
struct ProcessingConfig {
    int model_type;        // The type of click model (e.g., 0 = PBM).
    int node_id;           // The MPI communication rank of this node.
    int total_nodes;       // The number of nodes in the network.
    int thread_count;      // The number of CPU threads on this node.
    int* devices_per_node; // The number of devices per node in the network.
    int iterations;        // The number of iterations for which the EM algorithm should run.
    int exec_mode;         // The mode of execution (e.g., 0 = CPU, 1 = GPU).
    int device_count;      // The number of GPU devices on this node.
    int unit_count;        // The number of compute devices on this node (incuding CPU depending on the execution mode).
};

// LocalPartitions is a vector of tuples, where each tuple represents a
// a training set, a test set, and the number of query-document pairs assigned
// to a devices. The number of tuples in the vector is equal to the number of
// devices on the node.
using LocalPartitions = std::vector<std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>>;

/**
 * @brief Execute EM algorithm in parallel.
 *
 * @param config: Processing configuration.
 * @param device_partitions: The datasets assigned to the node's devices.
 * @param output_path: The path where to store the output.
 * @param hostname: The hostname of the device.
 */
 void em_parallel(
    const ProcessingConfig& config,
    LocalPartitions& device_partitions,
    const std::string& output_path,
    const char* hostname
);

#endif // CLICK_MODEL_PARALLEL_EM_H