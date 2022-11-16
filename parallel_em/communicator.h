/** MPI communication functions.
 *
 * communicate.h:
 *  - Defines several MPI functions used to communicate across different machines.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_MPI_H
#define CLICK_MODEL_MPI_H

// MPI include.
#ifndef __CUDACC__
    // Include MPI only on non-CUDA files.
    #include <mpi.h>
#endif

// System include.
#include <map>
#include <climits>
#include <iostream>
#include <fstream>

// User include.
#include "../utils/definitions.h"
#include "../utils/macros.cuh"
#include "../utils/utils.cuh"
#include "../click_models/param.cuh"
#include "../click_models/evaluation.h"
#include "../data/dataset.h"

struct QDP {
    int query;
    int document;
    float probability;
};

namespace Communicate {
    void initiate(int& argc, char**& argv, int& n_nodes, int& node_id);
    void finalize(void);
    void barrier(void);
    void get_n_devices(const int& n_devices, int* n_devices_network);
    void gather_properties(const int& node_id, const int& n_nodes, const int& n_devices, int* n_devices_network,
        std::vector<std::vector<std::vector<int>>>& network_properties, const int* device_architecture,
        const int* free_memory);
    void send_partitions(const int& node_id, const int& n_nodes, const int& n_devices, const int& total_n_devices,
        const int* n_devices_network, Dataset& dataset,
        std::vector<std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>>& device_partitions,
        std::vector<std::unordered_map<int, std::unordered_map<int, int>>*>& root_mapping);
    void exchange_parameters(std::vector<std::vector<std::vector<Param>>>& dest,
        const std::vector<std::vector<Param>>& my_params, const int n_nodes, const int node_id);
    void sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters);
    void gather_evaluations(std::map<int, std::array<float, 2>>& loglikelihood,
        std::map<int, Perplexity>& perplexity, const int n_nodes, const int node_id, const int* n_devices_network);
    void gather_results(const int node_id, const int target_id, std::vector<QDP>& parameters);
}


#endif // CLICK_MODEL_MPI_H