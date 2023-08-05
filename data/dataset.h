/** Click model dataset class.
 *
 * dataset.h:
 *  - Declare the Dataset class and functions using this class.
 */

#ifndef CLICK_MODEL_DATASET_H
#define CLICK_MODEL_DATASET_H

// System includes.
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <limits>
#include <functional>

// User includes.
#include "../utils/definitions.h"
#include "../utils/types.h"
#include "../utils/macros.cuh"
#include "search.cuh"
#include "../click_models/base.cuh"
#include "../parallel_em/communicator.h"
class ClickModel_Hst;

using node = int;
using device = int;
using distribution = double;
using avail_memory = size_t;
using query = double;
using DMap = std::unordered_map<int, int>;
using QDMap = std::unordered_map<int, DMap>;
using QueryDocumentMap = QDMap;

// The hash ring with nodes distributed according to the partitioning type.
class HashRing {
public:
    void init(const DeviceLayout1D<std::pair<distribution, avail_memory>>& layout, const ClusterProperties& cluster_properties, const ProcessingConfig& config);
    std::pair<node, device> get_destination(int key);
private:
    // The propeties of each destination.
    struct Destination {
        int node;                 // The node id of the destination.
        int device;               // The device id of the destination.
        size_t session_count = 0; // The number of sessions assigned to this destination.
        double distribution = 0;  // The portion of the hash ring assigned to this destination.
        double occupancy = 0;     // The amount of memory occupied by the sessions assigned to this destination.
        size_t free_memory = 0;   // The amount of free memory on this destination.
        Destination(int node, int device, size_t session_count, double distribution, double occupancy, size_t free_memory):
            node(node), device(device), session_count(session_count), distribution(distribution), occupancy(occupancy), free_memory(free_memory) {}
    };

    std::map<query, Destination> hash_ring; // The hash ring divided into ranges.
};

// Sort the partitions by query_id in ascending order
void sort_partitions(LocalPartitions& device_partitions, int n_threads);
// Parse the dataset in parallel on all nodes.
int parse_dataset(const ClusterProperties& cluster_properties, const ProcessingConfig& config, LocalPartitions& my_partitions);

#endif // CLICK_MODEL_DATASET_H
