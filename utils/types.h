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
using DeviceLayout = std::vector<std::vector<T>>; // Node ID -> Device ID -> Pointer.

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

#endif // TYPES_H