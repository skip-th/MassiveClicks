/** MPI communication functions.
 *
 * communicate.h:
 *  - Defines several MPI functions used to communicate across different machines.
 */

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
#include "../click_models/base.cuh"

namespace Communicate {
    void initiate(int& argc, char**& argv, int& n_nodes, int& node_id);
    void finalize(void);
    void error_check(std::string err_msg = "");
    void barrier(void);
    void get_n_devices(const int& n_devices, int* n_devices_network);
    void gather_properties(const int& node_id, const int& n_nodes, const int& n_devices, int* n_devices_network,
        std::vector<std::vector<std::vector<int>>>& network_properties, const int* device_architecture,
        const int* free_memory);
    void send_partitions(const int& node_id, const int& n_nodes, const int& n_devices, const int& total_n_devices,
        const int* n_devices_network, Dataset& dataset, LocalPartitions& device_partitions);
    void exchange_parameters(std::vector<std::vector<std::vector<Param>>>& dest,
        const std::vector<std::vector<Param>>& my_params, const int n_nodes);
    void sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters);
    void gather_evaluations(std::map<int, std::array<float, 2>>& loglikelihood,
        std::map<int, Perplexity>& perplexity, const int n_nodes, const int node_id, const int* n_devices_network);
    void output_parameters(const int node_id, const int processing_units, const std::string file_path,
        const LocalPartitions& dataset_partitions,
        const std::pair<std::vector<std::string>, std::vector<std::string>> &headers,
        const std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> *parameters);
}


#endif // CLICK_MODEL_MPI_H