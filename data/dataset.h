/** Click model dataset class.
 *
 * dataset.h:
 *  - Declare the Dataset class and functions using this class.
 */

#ifndef CLICK_MODEL_DATASET_H
#define CLICK_MODEL_DATASET_H

// Standard library includes.
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>

// Custom includes.
#include "../utils/definitions.h"
#include "../utils/types.h"
#include "../utils/macros.cuh"
#include "search.cuh"
#include "../click_models/base.cuh"

class ClickModel_Hst;

// The Dataset class encapsulates information related parsing the click log.
class Dataset {
public:
    Dataset();

    // Session-related methods.
    int get_num_sessions() const;
    void add_session(const int session_id, const UnassignedSet& session);

    // Query-related methods.
    int get_num_queries() const;
    void increment_num_queries(const int& value);
    void add_query_session(const SERP_Hst& query_session);

    // Training and testing set methods.
    int get_train_set_size(const int& node_id, const int& device_id) const;
    int get_test_set_size(const int& node_id, const int& device_id) const;
    int get_query_doc_pair_size(const int& node_id, const int& device_id) const;

    // Split data into training and testing sets.
    void split_data(const DeviceLayout<std::vector<int>>& network_properties, const float test_share, const int partitioning_type, const int model_type);

    // Accessors for training and testing sets.
    TrainSet* get_train_set(const int& node_id, const int& device_id);
    TestSet* get_test_set(const int& node_id, const int& device_id);

    // Retrieve the mapping of query-doc pairs.
    std::unordered_map<int, std::unordered_map<int, int>>* get_mapping(const int& node_id, const int& device_id);

private:
    // Partition data.
    void partition_data(const DeviceLayout<std::vector<int>>& network_properties, const float test_share, const int partitioning_type, const int model_type);

    // Helper method for reshaping.
    void reshape_pvar(const DeviceLayout<std::vector<int>>& network_properties);

    // Parameter addition methods for training and testing sets.
    void add_parameter_to_train_set(SERP_Hst& query_session, const int& node_id, const int& device_id);
    bool add_parameter_to_test_set(SERP_Hst& query_session, const int& node_id, const int& device_id);

    // Training set-related helper methods.
    std::pair<int,int> get_smallest_train_set(const DeviceLayout<TrainSet>& training_queries);
    std::pair<int,int> get_smallest_relative_train_set(const DeviceLayout<TrainSet>& training_queries, const DeviceLayout<std::vector<int>>& network_properties);
    std::pair<int,int> get_smallest_arch_train_set(const DeviceLayout<TrainSet>& training_queries, const DeviceLayout<std::vector<int>>& network_properties);

    // Device layout initialization method.
    template<typename A, typename B>
    void init_layout(DeviceLayout<A>& src, const DeviceLayout<B>& dst);

    // Member variables.
    int num_queries{0},  num_qd_pairs{0};
    ClickModel_Hst* click_model;

    UnassignedSet sessions;
    DeviceLayout<std::unordered_map<int, std::unordered_map<int, int>>> qd_parameters;
    DeviceLayout<int> qd_parameters_size;
    DeviceLayout<TrainSet> training_queries;
    DeviceLayout<TestSet> testing_queries;
};

// Utility functions.
int parse_dataset(Dataset &dataset, const std::string& raw_dataset_path, int max_sessions=-1);
void sort_partitions(LocalPartitions& device_partitions, int n_threads);

#endif // CLICK_MODEL_DATASET_H
