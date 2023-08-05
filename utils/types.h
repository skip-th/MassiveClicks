/** Type definitions.
 *
 * types.h:
 *  - Defines commonly used types, templates, and aliases used throughout
 *    the project.
 */

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include "../data/search.cuh"

// Type alias for the layout of devices in the cluster. The first dimension
// represents the node ID, and the second dimension represents the device ID.
// The value at each index is a pointer to the device.
template<typename T>
using DeviceLayout2D = std::vector<std::vector<T>>; // Node ID -> Device ID -> Value.
template<typename T>
using DeviceLayout1D = std::vector<std::tuple<int, int, T>>; // {{Node ID, Device ID, Value}, ...}.

// LocalPartitions is a vector of tuples, where each tuple represents a
// a training set, a test set, and the number of query-document pairs assigned
// to a devices. The number of tuples in the vector is equal to the number of
// devices on the node.
using UnassignedSet = std::vector<SERP_Hst>;
using TrainSet = UnassignedSet;
using TestSet = UnassignedSet;
using Partition = std::tuple<TrainSet, TestSet, int>;
using LocalPartitions = std::vector<Partition>;

// The properties of a device.
struct DeviceProperties {
    int device_id;
    int compute_capability; // e.g., 52 = 5.2
    int registers_per_block;
    int registers_per_sm;
    int threads_per_block;
    int threads_per_sm;
    int warp_size;
    int memory_clock_rate; // Hz
    int memory_bus_width; // bits
    int cores_per_sm;
    int clock_rate; // Hz
    int multiprocessor_count;
    size_t total_global_memory; // bytes
    size_t available_memory; // bytes
    size_t shared_memory_per_block; // bytes
    size_t total_constant_memory; // bytes
    size_t peak_performance; // Theoretical Peak FLOPs
    char device_name[256];
};

// The properties of a host.
struct HostProperties {
    int node_id;
    int device_count;
    int thread_count;
    size_t free_memory; // bytes
    size_t total_memory; // bytes
    char host_name[HOST_NAME_MAX];
};

// The properties of the node.
struct NodeProperties {
    HostProperties host;
    std::vector<DeviceProperties> devices;
};

// The properties of the cluster.
struct ClusterProperties {
    int node_count;
    int device_count;
    std::vector<NodeProperties> nodes;
};

// ProcessingConfig encapsulates various parameters needed for executing the
// EM algorithm in parallel. This includes identifiers for the model and node,
// the number of nodes, threads, iterations, and devices, the execution mode,
// and an information of other devices in the cluster.
struct ProcessingConfig {
    std::string dataset_path;      // The path to the raw click log.
    std::string output_path;       // The prefix of the output file.
    int job_id;                    // The job ID of the current execution.
    int max_sessions;              // The maximum number of sessions to read from the click log.
    int model_type;                // The type of click model (e.g., 0 = PBM).
    int partitioning_type;         // The type of partitioning (e.g., 0 = round-robin).
    int node_id;                   // The MPI communication rank of this node.
    int total_nodes;               // The number of nodes in the network.
    int thread_count;              // The number of CPU threads on this node.
    int* devices_per_node;         // The number of devices per node in the network.
    int iterations;                // The number of iterations for which the EM algorithm should run.
    int exec_mode;                 // The mode of execution (e.g., 0 = CPU, 1 = GPU).
    float test_share;              // The percentage of sessions to use for testing.
    int device_count;              // The number of GPU devices on this node.
    int max_gpus;                  // The maximum number of GPUs to use per node.
    int worker_count;              // The number of usable workers on this node (i.e., the number of GPUs).
    bool help;                     // Whether to print the help message.
    char host_name[HOST_NAME_MAX]; // The name of the current host.
};

#endif // TYPES_H