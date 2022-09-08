/** Improved implementation of parsing a given dataset.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * dataset.h:
 *  - Declare the Dataset class and functions using this class.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_DATASET_H
#define CLICK_MODEL_DATASET_H

// System include.
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>

// User include.
#include "../utils/definitions.h"
#include "search.cuh"

class Dataset {
public:
    Dataset();
    int size_sessions(void) const;
    int size_queries(void) const;
    int size_train(const int& nid, const int& did) const;
    int size_test(const int& nid, const int& did) const;
    int size_qd(const int& nid, const int& did) const;
    void increment_queries(const int& value);
    void add_session(const int session_id, const std::vector<SERP>& session);
    void add_query_session(const SERP& query_session);
    void make_splits(const NetworkMap<std::vector<int>>& network_properties, const float test_share, const int partitioning_type);
    std::vector<SERP>* get_train_set(const int& nid, const int& did);
    std::vector<SERP>* get_test_set(const int& nid, const int& did);
    std::unordered_map<int, std::unordered_map<int, int>>* get_mapping(const int& nid, const int& did);

private:
    void make_partitions(const NetworkMap<std::vector<int>>& network_properties, const float test_share, const int partitioning_type);
    void reshape_pvar(const NetworkMap<std::vector<int>>& network_properties);
    void add_parameter_train(SERP& query_session, const int& node_id, const int& device_id);
    bool add_parameter_test(SERP& query_session, const int& node_id, const int& device_id);
    std::pair<int,int> get_smallest_train(const NetworkMap<std::vector<SERP>>& training_queries);
    std::pair<int,int> get_smallest_relative_train(const NetworkMap<std::vector<SERP>>& training_queries, const NetworkMap<std::vector<int>>& network_properties);

    int n_queries{0};
    int n_qd_pairs{0};

    // std::unordered_map<int, std::vector<SERP>> sessions; // Session ID -> Query sessions with this session ID.
    std::vector<SERP> sessions;
    // A multi-dimensional array imitating the layout of the nodes and devices in the network.
    // Each node's device array contains a from the query-document pair to a unique index in the parameter array.
    NetworkMap<std::unordered_map<int, std::unordered_map<int, int>>> qd_parameters; // Node ID -> Device ID -> Query ID -> Document ID -> QD-pair index.
    // A multi-dimensional array imitating the layout of the nodes and devices in the network.
    // Each node's device array contains the number of query-document pairs represented by the qd_parameters mapping.
    NetworkMap<int> qd_parameters_sz; // Node ID -> Device ID -> Number of QD-pairs.
    // A multi-dimensional array imitating the layout of the nodes and devices in the network.
    // Each node's device array is a 1D array of contiguous query sessions with the same query ID.
    NetworkMap<std::vector<SERP>> training_queries; // Node ID -> Device ID -> SERP's grouped by query.
    // A multi-dimensional array imitating the layout of the nodes and devices in the network.
    // Each node's device array is a 1D array of contiguous query sessions with the same session ID,
    // whose query ID occured on the same node's device in the training set.
    NetworkMap<std::vector<SERP>> testing_queries; // Node ID -> Device ID -> SERP's grouped by session.

    // Utility function used to reshape a multi-dimensional array to the same
    // dimensions as another array.
    template<typename A, typename B>
    void init_network(NetworkMap<A>& src, const NetworkMap<B>& dst) {
        src.resize(dst.size());
        for (int nid = 0; nid < dst.size(); nid++) { src[nid].resize(dst[nid].size()); }
    }
};

void parse_dataset(Dataset &dataset, const std::string& raw_dataset_path, int max_sessions=-1);

#endif // CLICK_MODEL_DATASET_H