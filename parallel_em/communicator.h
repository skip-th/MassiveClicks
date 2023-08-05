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
#include <string>

// User include.
#include "../utils/definitions.h"
#include "../utils/macros.cuh"
#include "../utils/utils.cuh"
#include "../click_models/param.cuh"
#include "../click_models/evaluation.h"
#include "../data/dataset.h"
#include "../click_models/base.cuh"

namespace Communicate {
    // MPI Datatypes
    extern MPI_Datatype MPI_SERP;
    extern MPI_Datatype MPI_HST_PROP;
    extern MPI_Datatype MPI_DEV_PROP;
    extern MPI_Datatype MPI_PARAM;
    extern MPI_Datatype MPI_LLH;
    extern MPI_Datatype MPI_PPL;

    // Initiates the communication module.
    void initiate(int& argc, char**& argv, int& n_nodes, int& node_id);

    // Finalizes the communication module.
    void finalize();

    // Performs error checking, with optional error message.
    void error_check(std::string err_msg = "");

    // Synchronizes all processes to a point.
    void barrier();

    // Gathers properties from all nodes.
    ClusterProperties gather_properties(NodeProperties local_node_properties);

    // Sends sessions to a specific destination.
    void send_sessions(const int dst, int device, SERP_Hst& session);

    // Receives sessions from a specific source.
    int recv_sessions(const int src, LocalPartitions& my_partitions);

    // Get the properties of the datasets on each node.
    void gather_partition_properties(ClusterProperties& cluster_properties,
                                     ProcessingConfig& config,
                                     DeviceLayout2D<std::tuple<int,int,int>>& layout,
                                     LocalPartitions& my_partitions);

    // Exchanges parameters between nodes.
    void exchange_parameters(std::vector<std::vector<std::vector<Param>>>& dest,
                             const std::vector<std::vector<Param>>& my_params);

    // Synchronizes parameters across all nodes and their devices.
    void sync_parameters(DeviceLayout2D<std::vector<Param>>& parameters);

    // Gathers evaluation results from all nodes.
    void gather_evaluations(std::map<int, std::array<float, 2>>& loglikelihood,
                            std::map<int, Perplexity>& perplexity,
                            const int* n_devices_network);

    // Outputs parameters to separate files.
    void output_parameters(const int workers,
                           const std::string file_path,
                           const LocalPartitions& dataset_partitions,
                           const std::pair<std::vector<std::string>, std::vector<std::string>> &headers,
                           const std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> *parameters);
}

#endif // CLICK_MODEL_MPI_H