/** Click model dataset class.
 *
 * dataset.cpp:
 *  - Defines the behaviour of the Dataset class.
 *  - Parses a given dataset and assembles it into the Dataset class.
 */

#include "dataset.h"

//---------------------------------------------------------------------------//
// Dataset                                                                   //
//---------------------------------------------------------------------------//

/**
 * @brief Constructor for the Dataset class.
 */
Dataset::Dataset() = default;

/**
 * @brief The number of sessions present in the parsed dataset.
 *
 * @return The number of sessions present in the parsed dataset.
 */
int Dataset::get_num_sessions(void) const{
    return this->sessions.size();
}

/**
 * @brief Add the query sessions contained within a single session to the dataset.
 *
 * @param session_id The session id shared by all provided query sessions.
 * @param session The session containing the query sessions.
 */
void Dataset::add_session(const int session_id, const UnassignedSet& session) {
    for (auto serp : session) {
        // this->sessions[session_id].push_back(serp);
        this->sessions.push_back(serp);
    }
}

/**
 * @brief The number of queries present in the parsed dataset.
 *
 * @return The number of queries present in the parsed dataset.
 */
int Dataset::get_num_queries(void) const{
    return this->num_queries;
}

/**
 * @brief Increments the number of queries in the dataset by a given amount.
 *
 * @param value The value by which to increment the number of queries.
 */
void Dataset::increment_num_queries(const int& value) {
    this->num_queries += value;
}

/**
 * @brief Add the given query session to the dataset.
 *
 * @param query_session The query session which will be added.
 */
void Dataset::add_query_session(const SERP_Hst& query_session) {
    // this->sessions[session_id].push_back(query_session);
    this->sessions.push_back(query_session);
}

/**
 * @brief The number of training query sessions assigned to a node's device.
 *
 * @param node_id The id of the node.
 * @param device_id The id of the device on the node.
 * @return The number of training query sessions assigned to the node's device.
 */
int Dataset::get_train_set_size(const int& nid, const int& did) const{
    return this->training_queries[nid][did].size();
}

/**
 * @brief The number of testing query sessions assigned to a node's device.
 *
 * @param node_id The id of the node.
 * @param device_id The id of the device on the node.
 * @return The number of training query sessions assigned to the node's device.
 */
int Dataset::get_test_set_size(const int& nid, const int& did) const{
    return this->testing_queries[nid][did].size();
}

/**
 * @brief The number of unique query-document pairs assigned to a node's device.
 *
 * @param node_id The id of the node.
 * @param device_id The id of the device on the node.
 * @return The number of unique query-document pairs assigned to a node's device.
 */
int Dataset::get_query_doc_pair_size(const int& nid, const int& did) const{
    return this->qd_parameters_size[nid][did];
}

/**
 * @brief Retrieves the training queries assigned to a node's device.
 *
 * @param node_id The id of the node.
 * @param device_id The id of the device on the node.
 * @return A reference to the training queries vector.
 */
TrainSet* Dataset::get_train_set(const int& nid, const int& did) {
    return &this->training_queries[nid][did];
}

/**
 * @brief Retrieves the testing queries assigned to a node's device.
 *
 * @param node_id The id of the node.
 * @param device_id The id of the device on the node.
 * @return A reference to the testing queries vector.
 */
TestSet* Dataset::get_test_set(const int& nid, const int& did) {
    return &this->testing_queries[nid][did];
}

/**
 * @brief Retrieves the mapping of query-document pairs to parameter indices.
 *
 * @param node_id The id of the node.
 * @param device_id The id of the device on the node.
 * @return A reference to the query-document pair mapping.
 */
std::unordered_map<int, std::unordered_map<int, int>>* Dataset::get_mapping(const int& nid, const int& did) {
    return &this->qd_parameters[nid][did];
}

/**
 * @brief Adds the index of parameter associated with a query-document pair to
 * the search results in the given query session, if the query-document pair
 * has been encountered before when adding the training parameter indices.
 *
 * @param query_session The query session containing the documents to which the
 * parameter idices will be added.
 * @param node_id The id of the node.
 * @param device_id The id of the device on the node.
 * @return true, if all query-document pairs were successfully assigned a
 * parameter index, false otherwise.
 */
bool Dataset::add_parameter_to_test_set(SERP_Hst& query_session, const int& node_id, const int& device_id) {
    std::unordered_map<int, std::unordered_map<int, int>>* local_params = &this->qd_parameters[node_id][device_id];

    // Iterate over all ranks in the query session.
    std::unordered_map<int, std::unordered_map<int, int>>::iterator qitr;
    bool found_match = false;
    for (int rank = 0; rank < MAX_SERP; rank++) {
        qitr = local_params->find(query_session.get_query());

        // If the query exists in the map, check if the document exists.
        if (qitr != local_params->end()) {
            std::unordered_map<int, int>::iterator ditr = qitr->second.find(query_session.access_sr(rank).get_doc_id());

            // If the document exists, add the parameter index to the search result.
            if (ditr != qitr->second.end()) {
                query_session.access_sr(rank).set_param_index(ditr->second);
                found_match = true;
            }
        }
    }

    // Return true if at least one of the documents inside the query session
    // occurs within the training set of the current device.
    return found_match;
}

/**
 * @brief Adds the index of parameter associated with a query-document pair to
 * the search results in the given query session.
 *
 * @param query_session The query session containing the documents to which the
 * parameter idices will be added.
 * @param node_id The id of the node.
 * @param device_id The id of the device on the node.
 */
void Dataset::add_parameter_to_train_set(SERP_Hst& query_session, const int& node_id, const int& device_id) {
    std::unordered_map<int, std::unordered_map<int, int>>* local_params = &this->qd_parameters[node_id][device_id];
    int* local_params_sz = &this->qd_parameters_size[node_id][device_id];
    int query = query_session.get_query();

    // Iterate over all ranks in the query session.
    std::unordered_map<int, std::unordered_map<int, int>>::iterator qitr;
    for (int rank = 0; rank < MAX_SERP; rank++) {
        qitr = local_params->find(query);

        // If the query exists in the map, check if the document exists.
        if (qitr != local_params->end()) {
            std::unordered_map<int, int>::iterator ditr = qitr->second.find(query_session[rank].get_doc_id());

            // If the document exists, add the parameter index to the search result.
            if (ditr != qitr->second.end()) {
                query_session.access_sr(rank).set_param_index(ditr->second);
            }
            // If it doesn't exist, then add the parameter index to the search result and map.
            else {
                qitr->second[query_session[rank].get_doc_id()] = *local_params_sz;
                query_session.access_sr(rank).set_param_index(*local_params_sz);
                (*local_params_sz)++; // Increase the size of this device's qd-parameters.
            }
        }
        // If the query doesn't exists in the map, add the parameter index to the search result and map.
        else {
            // Add a new entry to the map for this qd-parameter.
            (*local_params)[query][query_session[rank].get_doc_id()] = *local_params_sz;
            // Add the parameter index to the search result.
            query_session.access_sr(rank).set_param_index(*local_params_sz);
            // Increase the size of this device's qd-parameters.
            (*local_params_sz)++;
        }
    }
}

/**
 * @brief Finds the device with the least filled training query vector.
 *
 *
 * @param training_queries A vector containing SERP_Hsts grouped by query.
 *
 * @return The node id and device id of smallest training vector.
 */
std::pair<int,int> Dataset::get_smallest_train_set(const DeviceLayout<TrainSet>& training_queries) {
    int smallest{std::numeric_limits<int>::max()}, small_nid{0}, small_did{0};

    for (size_t nid = 0; nid < training_queries.size(); nid++) {
        for (size_t did = 0; did < training_queries[nid].size(); did++) {
            if (training_queries[nid][did].size() <= static_cast<size_t>(smallest)) {
                smallest = training_queries[nid][did].size();
                small_nid = nid;
                small_did = did;
            }
        }
    }

    return std::make_pair(small_nid, small_did);
}

/**
 * @brief Finds the device with the least filled training query vector relative
 * to the available memory on the device.
 *
 * @param training_queries A vector containing SERP_Hsts grouped by query.
 * @param network_properties The properties of the devices within the network.
 *
 * @return The node id and device id of smallest training
 * vector relative to memory size.
 */
std::pair<int,int> Dataset::get_smallest_relative_train_set(const DeviceLayout<TrainSet>& training_queries, const DeviceLayout<std::vector<int>>& network_properties) {
    int small_nid{0}, small_did{0};
    float smallest{std::numeric_limits<float>::max()}, occupancy{0};

    for (size_t nid = 0; nid < training_queries.size(); nid++) {
        for (size_t did = 0; did < training_queries[nid].size(); did++) {
            occupancy = (float) training_queries[nid][did].size() / (float) network_properties[nid][did][1];
            if (occupancy <= smallest) {
                smallest = occupancy;
                small_nid = nid;
                small_did = did;
            }
        }
    }

    return std::make_pair(small_nid, small_did);
}

/**
 * @brief Finds the device with the newest architecture and fills its training
 * query vector first.
 *
 * @param training_queries A vector containing SERP_Hsts grouped by query.
 * @param network_properties The properties of the devices within the network.
 *
 * @return The node id and device id of smallest training
 * vector with the highest device architecture.
 */
std::pair<int,int> Dataset::get_smallest_arch_train_set(const DeviceLayout<TrainSet>& training_queries, const DeviceLayout<std::vector<int>>& network_properties) {
    int small_new_nid{0}, small_new_did{0}, arch, prev_arch{0};
    float memory_footprint, occupancy, smallest_occupancy{std::numeric_limits<float>::max()};

    for (size_t nid = 0; nid < training_queries.size(); nid++) {
        for (size_t did = 0; did < training_queries[nid].size(); did++) {
            memory_footprint = click_model->compute_memory_footprint(training_queries[nid][did].size(), this->qd_parameters_size[nid][did]) / 1e6;
            // Calculate the occupancy of the device with a margin of error of 0.1%.
            occupancy = memory_footprint * 1.001 / (float) network_properties[nid][did][1];

            // If the device is newer than the previous device and is less
            // occupied than the other devices with the same architecture, then
            // make it the new target. Ignore the device if it is full.
            if (occupancy < smallest_occupancy && occupancy < 1.f) { // Use "if (occupancy < 1.f) {" to fill up one new arch device first.
                arch = network_properties[nid][did][0];

                if (arch >= prev_arch) {
                    smallest_occupancy = occupancy;
                    small_new_nid = nid;
                    small_new_did = did;
                    prev_arch = arch;
                }
            }
        }
    }

    return std::make_pair(small_new_nid, small_new_did);
}

/**
 * @brief Finds the device with the least filled training query vector relative
 * to the number of CUDA cores on the device.
 *
 * @param training_queries A vector containing SERP_Hsts grouped by query.
 * @param cluster_properties The properties of the devices within the network.
 *
 * @return std::pair<int,int>
 */
std::pair<int,int> Dataset::get_smallest_core_train_set(const DeviceLayout<TrainSet>& training_queries, const ClusterProperties& cluster_properties) {
    int small_nid{0}, small_did{0};
    float smallest{std::numeric_limits<float>::max()}, occupancy{0};

    for (size_t nid = 0; nid < training_queries.size(); nid++) {
        for (size_t did = 0; did < training_queries[nid].size(); did++) {
            occupancy = (float) training_queries[nid][did].size() / (float) (cluster_properties.nodes[nid].devices[did].cores_per_sm * cluster_properties.nodes[nid].devices[did].multiprocessor_count);
            if (occupancy <= smallest) {
                smallest = occupancy;
                small_nid = nid;
                small_did = did;
            }
        }
    }

    return std::make_pair(small_nid, small_did);
}

/**
 * @brief Finds the device with the least filled training query vector relative
 * to its theoretical peak performance.
 *
 * @param training_queries A vector containing SERP_Hsts grouped by query.
 * @param cluster_properties The properties of the devices within the network.
 *
 * @return std::pair<int,int>
 */
std::pair<int,int> Dataset::get_smallest_perf_train_set(const DeviceLayout<TrainSet>& training_queries, const ClusterProperties& cluster_properties) {
    int small_nid{0}, small_did{0};
    float smallest{std::numeric_limits<float>::max()}, occupancy{0};

    for (size_t nid = 0; nid < training_queries.size(); nid++) {
        for (size_t did = 0; did < training_queries[nid].size(); did++) {
            occupancy = (float) training_queries[nid][did].size() / (float) cluster_properties.nodes[nid].devices[did].peak_performance;
            if (occupancy <= smallest) {
                smallest = occupancy;
                small_nid = nid;
                small_did = did;
            }
        }
    }

    return std::make_pair(small_nid, small_did);
}

/**
 * @brief Partitions the parsed dataset into training and testing sets
 * according to a given partitioning scheme. The resulting sets are assigned
 * to the available devices on different nodes.
 *
 * @param network_properties The properties of the devices within the network.
 * @param test_share The share of the dataset that will be used for testing.
 * @param partitioning_type The partitioning scheme to use (e.g., Round-Robin).
 * @param model_type The type of click model to measure the memory footprint
 * with (e.g., 0 = PBM, 1 = CCM).
 */
void Dataset::partition_data(const ClusterProperties cluster_properties, const DeviceLayout<std::vector<int>>& network_properties, const float test_share, const int partitioning_type, const int model_type) {
    int node_id{0}, device_id{0}, n_nodes = static_cast<int>(network_properties.size());
    this->click_model = create_cm_host(model_type);

    // Calculate the length of the training set using the test share. The
    // training set length is rounded to the nearest integer.
    int n_train{static_cast<int>(this->num_queries * (1 - test_share) - 0.5) + 1};

    std::cout << "\nPartitioning " << this->num_queries << " queries into " <<
    n_train << " training and " << this->num_queries - n_train <<
    " testing sessions.\n(1/3) Grouping training query sessions..." << std::endl;

    // Group the indices of the query sessions inside the sessions designated
    // for training by their query.
    std::unordered_map<int, std::vector<int>> grouped_train_queries; // Query ID -> [Session ID, Subsession index]
    std::unordered_map<int, std::vector<int>> grouped_test_queries; // Query ID -> [Session ID, Subsession index]
    int ses_index{0};
    while (ses_index < this->num_queries) {
        if (ses_index < n_train) {
            grouped_train_queries[this->sessions[ses_index].get_query()].push_back(ses_index);
        }
        else {
            grouped_test_queries[this->sessions[ses_index].get_query()].push_back(ses_index);
        }
        ses_index++;
    }

    // Assign training query sessions to the training array of a node's device.
    std::cout << "(2/3) Partitioning training query sessions..." << std::endl;
    std::unordered_map<int, std::vector<int>>::iterator grp_itr = std::begin(grouped_train_queries);
    for (; grp_itr != std::end(grouped_train_queries); grp_itr++, device_id++) {
        // Get the next training set according to the partitioning type.
        switch (partitioning_type) {
            case 0: // Round-Robin
            default: // Round-Robin is the default
                // Get the next device from the corresponding node in Round-Robin fashion.
                if (device_id >= static_cast<int>(this->training_queries[node_id].size())) {
                    node_id = (node_id + 1) % n_nodes;
                    device_id = 0;
                }
                break;
            case 1: // Maximum Utilization
                // Get the node id and device id of where the smallest training set exists.
                std::tie(node_id, device_id) = this->get_smallest_train_set(this->training_queries);
                break;
            case 2: // Proportional Maximum Utilization
                // Get the node id and device id of where the smallest training set relative to total device memory exists.
                std::tie(node_id, device_id) = this->get_smallest_relative_train_set(this->training_queries, network_properties);
                break;
            case 3: // Newest Architecture First
                // Get the node id and device id of where the smallest training set relative to the newest architecture exists.
                std::tie(node_id, device_id) = this->get_smallest_arch_train_set(this->training_queries, network_properties);
                break;
            case 4: // Relative CUDA cores
                // Get the node id and device id of where the smallest training set relative to the number of CUDA cores exists.
                std::tie(node_id, device_id) = this->get_smallest_core_train_set(this->training_queries, cluster_properties);
                break;
            case 5: // Relative peak performance
                // Get the node id and device id of where the smallest training set relative to the device's peak performance exists.
                std::tie(node_id, device_id) = this->get_smallest_perf_train_set(this->training_queries, cluster_properties);
                break;
        }

        // For each document in a query group's query session, get the
        // device-local parameter index from qd_parameters and assign the query
        // session to the training array of the device.
        for (int ses_i : grp_itr->second) {
            SERP_Hst serp = this->sessions[ses_i];
            this->add_parameter_to_train_set(serp, node_id, device_id);
            this->training_queries[node_id][device_id].push_back(serp);
        }
    }

    // Move the remaining sessions to the testing set.
    // Check for each SERP_Hst from the test sessions whether all the SERP_Hst's search
    // results occur in a device's qd_parameters.
    std::cout << "(3/3) Partitioning testing query sessions..." << std::endl;
    for (grp_itr = std::begin(grouped_test_queries); grp_itr != std::end(grouped_test_queries); grp_itr++) {
        // Go through all queries in the current session.
        for (int ses_i : grp_itr->second) {
            SERP_Hst qry = this->sessions[ses_i];

            // Go through all devices in the network.
            for (int nid = 0; nid < n_nodes; nid++) {
                for (int did = 0; did < static_cast<int>(this->qd_parameters[nid].size()); did++) {
                    // Check if all query-document pairs from this SERP_Hst exist
                    // in the qd_parameters of this device.
                    if (this->add_parameter_to_test_set(qry, nid, did)) {
                        // Add the query session to the device's testing set.
                        this->testing_queries[nid][did].push_back(qry);
                        goto next_qry;
                    }
                }
            }

            next_qry:;
        }
    }
}

/**
 * @brief Initializes the given multi-dimensional vectors by matching their
 * shape to the given device layout.
 *
 * @tparam A The type of the source vector.
 * @tparam B The type of the destination vector.
 * @param src The source vector.
 * @param dst The destination vector.
 */
template<typename A, typename B>
void Dataset::init_layout(DeviceLayout<A>& src, const DeviceLayout<B>& dst) {
    src.resize(dst.size());
    for (size_t node_id = 0; node_id < dst.size(); node_id++) {
        src[node_id].resize(dst[node_id].size());
    }
}

/**
 * @brief Initializes the given multi-dimensional vectors by matching their
 * shape to the given network map.
 *
 * @param network_properties The network map containing the properties of the
 * devices within the network.
 */
void Dataset::reshape_pvar(const DeviceLayout<std::vector<int>>& network_properties) {
    this->init_layout(this->training_queries, network_properties);
    this->init_layout(this->testing_queries, network_properties);
    this->init_layout(this->qd_parameters, network_properties);
    this->init_layout(this->qd_parameters_size, network_properties);

    // Initialize qd parameter size at 0.
    for (size_t nid = 0; nid < network_properties.size(); nid++) {
        for (size_t did = 0; did < network_properties[nid].size(); did++) {
            this->qd_parameters_size[nid][did] = 0;
        }
    }
}

/**
 * @brief Split the parsed dataset into training and testing sets and partition
 * the result among the devices within the network using the given partitioning
 * scheme.
 *
 * @param network_properties The properties of the devices within the network.
 * @param test_share The share of the dataset that will be used for testing.
 * @param partitioning_type The partitioning scheme to use (e.g., Round-Robin).
 * @param model_type The type of click model to measure the memory footprint
 * with (e.g., 0 = PBM, 1 = CCM).
 */
void Dataset::split_data(const ClusterProperties cluster_properties, const DeviceLayout<std::vector<int>>& network_properties, const float test_share, const int partitioning_type, const int model_type) {
    // Shape the multi-dimensional arrays training, testing and parameter arrays
    // according to the network topology beforehand.
    this->reshape_pvar(network_properties);

    // Assign query groups from the dataset to different partitions according to
    // the chosen partitioning approach.
    partition_data(cluster_properties, network_properties, test_share, partitioning_type, model_type);
}


//---------------------------------------------------------------------------//
// Sort dataset partition                                                    //
//---------------------------------------------------------------------------//

/**
 * @brief Sort a device's training set by query id using quicksort.
 *
 * @param device_partitions The partitions to sort.
 * @param n_threads The number of threads to available.
 */
void sort_partitions(LocalPartitions& device_partitions, int n_threads) {
    auto sort_partition = [](UnassignedSet& partition) {
        std::sort(partition.begin(), partition.end(),
                [](const SERP_Hst& a, const SERP_Hst& b) { return a.get_query() < b.get_query(); });
    };

    // Check whether there are enough threads to sort the partitions in parallel.
    size_t n_partitions = device_partitions.size();
    if (static_cast<size_t>(n_threads) >= n_partitions) {
        // Sort the partitions in parallel.
        std::thread threads[n_partitions];
        for (size_t i = 0; i < n_partitions; i++) {
            threads[i] = std::thread(sort_partition, std::ref(std::get<0>(device_partitions[i])));
        }

        // Wait for all threads to finish.
        for (size_t i = 0; i < n_partitions; i++) {
            threads[i].join();
        }
    }
    else {
        // Sort the partitions sequentially.
        for (size_t did = 0; did < n_partitions; did++) {
            sort_partition(std::get<0>(device_partitions[did]));
        }
    }
}


//---------------------------------------------------------------------------//
// Parse raw dataset                                                         //
//---------------------------------------------------------------------------//

/**
 * @brief Parses a raw dataset file into a vector where each query session is
 * grouped by its session id.
 *
 * @param dataset The dataset class which will contain the parsed results.
 * @param raw_dataset_path The path to the raw dataset text file.
 * @param max_sessions The maximum number of sessions to parse. This does not
 * equal the number of lines in the file.
 *
 * @return Success of the parsing operation. (0 = success, 1 = failure)
 */
int parse_dataset(Dataset &dataset, const std::string& raw_dataset_path, int max_sessions) {
    // Open an input file stream using the given dataset path.
    std::ifstream raw_file(raw_dataset_path);
    std::string line;
    std::string element;
    int n_lines{0}, n_queries{0}, progress{0};

    // Proceed to reading the dataset if the input stream has succesfully been opened.
    if (raw_file.is_open()) {
        std::cout << "Raw dataset file \"" << raw_dataset_path << "\" is opened." << std::endl;

        SERP_Hst curr_SERP_Hst;
        int session_id_curr{0}, session_id_prev{-1};

        // Extract lines of type string from the dataset input stream, while the
        // number of extracted lines does not exceed the indicated maximum number of sessions.
        // The last session is not added to the dataset due to sessions only being
        // added when a new session id is encountered.
        while (std::getline(raw_file, line) && (n_queries < max_sessions || max_sessions == -1)) {
            // Create a stream for the line string and a string vector to store the
            // processed string.
            std::stringstream ssi(line);
            std::vector<std::string> line_vec;

            // Only add an element from the input stream to the current entire
            // line string if the element is errorless.
            while (ssi.good()) {
                ssi >> element;
                line_vec.push_back(element);
            }
            session_id_curr = std::stoi(line_vec[0]);
            n_lines++;

            // If this line contains 15 elements, then it describes a query action.
            // Vector elements: session id, time passed, Q, query id, region id,
            // document id (rank = 1), ..., document id (rank = MAX_SERP).
            if (line_vec.size() == QUERY_LINE_LENGTH) {
                // Add the previous query to the parsed dataset. Ensure that
                // this query session has actually been filled by checking
                // whether the session id does not equal its default value (-1).
                if (session_id_prev != -1) {
                    dataset.add_query_session(curr_SERP_Hst);
                    n_queries++;
                }

                curr_SERP_Hst = SERP_Hst(line_vec);
            }
            // If this line contains 4 elements, then it describes a click action.
            // Vector elements: session id, time passed, C, document id.
            else if (line_vec.size() == CLICK_LINE_LENGTH) {
                curr_SERP_Hst.update_click_res(line_vec);
            }
            // If this line contains any other number of elements, then the
            // associated action type is unknown.
            else {
                std::cout << "\nInvalid data: " << line << std::endl;
            }

            session_id_prev = session_id_curr;

            // Show progress.
            int percent = static_cast<int>((static_cast<float>(n_queries) * 100.f) / max_sessions);
            if (percent != progress) {
                std::cout << "\r[" << std::string(percent / 5, '=') << std::string(20 - percent / 5, ' ') << "] " << percent << "%" << std::flush;
                progress = percent;
            }
        }
    }
    else {
        return 1;
    }

    dataset.increment_num_queries(n_queries);
    std::cout << "\rRead " << n_lines << " lines." << std::string(15, ' ') << std::endl;
    return 0;
}
