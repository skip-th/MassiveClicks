/** Click model dataset class.
 *
 * dataset.cpp:
 *  - Defines the behaviour of the Dataset class.
 *  - Parses a given dataset and assembles it into the Dataset class.
 */

#include "dataset.h"

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

/**
 * @brief Initializes the hash ring with the given layout.
 *
 * @param layout The layout of the hash ring including the node id, device
 * id, portion of the hash ring assigned to each destination, and the available
 * memory of each destination.
 * @param cluster_properties The properties of the devices within the network.
 * @param config The processing configuration.
 */
void HashRing::init(const DeviceLayout1D<std::pair<distribution, avail_memory>>& layout, const ClusterProperties& cluster_properties, const ProcessingConfig& config) {
    this->hash_ring.clear(); // Clear the hash ring.
    double range_end = 0.0;

    for (const auto& properties : layout) { // Set a range for each destination.
        // Set destination properties.
        Destination dest = {std::get<0>(properties), std::get<1>(properties), 0, std::get<2>(properties).first, 0, std::get<2>(properties).second};
        range_end += dest.distribution;

        // Set destination range.
        if (!this->hash_ring.emplace(range_end, dest).second) { // Check whether a destination with the same range start already exists.
            const char* root_name = cluster_properties.nodes[config.node_id].host.host_name;
            const char* host_name = cluster_properties.nodes[std::get<0>(properties)].host.host_name;
            std::string device_name = (config.exec_mode == 0 || config.exec_mode == 2) ? "'s " + std::string(cluster_properties.nodes[std::get<0>(properties)].devices[std::get<1>(properties)].device_name) : "";
            std::cout << "[" << root_name << "] \033[12;33mWarning\033[0m: Distribution of " << std::get<2>(properties).first * 100 << "\% of node " << host_name << device_name << " is too small! No sessions allocated." << std::endl;
        }
    }
}

/**
 * @brief Get the destination node and device.
 *
 * @param key The query id to hash.
 * @return std::pair<node, device> The destination node and device.
 */
std::pair<node, device> HashRing::get_destination(int key) {
    std::size_t hashed_key = std::hash<std::string>{}(std::to_string(key));
    double normalized_hash = static_cast<double>(hashed_key) / std::numeric_limits<std::hash<std::string>::result_type>::max();

    // Find the destination whose range start is less than or equal to the key's normalized hash.
    auto it = this->hash_ring.lower_bound(normalized_hash);
    it->second.session_count++; // Increase session count of destination.
    return std::make_pair(it->second.node, it->second.device);
}

/**
 * @brief Create a 1D layout of a property corresponding to the devices on each
 * node.
 *
 * @tparam PropertyType The type of the property to get.
 * @tparam Function The type of the function to get the property.
 * @param layout The layout to fill with node id, device id and property.
 * @param cluster_properties The properties of the devices within the network.
 * @param get_property The function to get the property.
 */
template <typename PropertyType, typename Function>
void get_property_layout(DeviceLayout1D<PropertyType>& layout, const ClusterProperties& prop, const ProcessingConfig& config,  Function get_property) {
    for (auto node : prop.nodes) {
        if (config.exec_mode == 0 || config.exec_mode == 2) { // GPU-only or Hybrid
            for (int device_id = 0; device_id < node.devices.size(); device_id++) {
                layout.push_back({node.host.node_id, device_id, get_property(prop, config, node.host, node.host.node_id, device_id)});
            }
        }
        else if (config.exec_mode == 1) { // CPU-only
            layout.push_back({node.host.node_id, 0, get_property(prop, config, node.host, node.host.node_id, -1)});
        }
    }
}

/**
 * @brief Get the available memory of a device.
 *
 * @param cluster_properties The properties of the devices within the network.
 * @param config The processing configuration.
 * @param node_id The node id of the device.
 * @param device_id The device id of the device.
 * @return size_t The available memory of the device.
 */
size_t get_available_memory(const ClusterProperties& cluster_properties, const ProcessingConfig& config, const int node_id, const int device_id) {
    size_t total_host_memory = cluster_properties.nodes[node_id].host.free_memory;
    if (config.exec_mode == 0 || config.exec_mode == 2) { // GPU-only or Hybrid
        size_t total_device_memory = std::accumulate(cluster_properties.nodes[node_id].devices.begin(), cluster_properties.nodes[node_id].devices.end(), 0UL, [](size_t sum, DeviceProperties device) {return sum + device.available_memory;});
        // Take the lesser of the two memories, host and device, to ensure that the dataset will fit.
        return std::min(static_cast<size_t>(total_host_memory * static_cast<double>(cluster_properties.nodes[node_id].devices[device_id].available_memory) / static_cast<double>(total_device_memory)),
                        cluster_properties.nodes[node_id].devices[device_id].available_memory);
    }
    return total_host_memory; // CPU-only
}

/**
 * @brief Set the partitioning policy to initialize the hash ring with.
 *
 * @param cluster_properties The properties of the devices within the network.
 * @param config The processing configuration.
 * @param hash_ring The hash ring to initialize.
 */
void set_partitioning_policy(const ClusterProperties& cluster_properties, const ProcessingConfig& config, HashRing& hash_ring) {
    // Lambda to normalize the layout values.
    auto normalizeDeviceLayout1D = [](const DeviceLayout1D<std::pair<double, size_t>>& layout) {
        double total_value = std::accumulate(layout.begin(), layout.end(), 0UL, [](double sum, std::tuple<int,int,std::pair<double, size_t>> layout) {return sum + std::get<2>(layout).first;});
        DeviceLayout1D<std::pair<double,size_t>> normalized_layout;
        for (size_t i = 0; i < layout.size(); i++)
            normalized_layout.push_back({std::get<0>(layout[i]), std::get<1>(layout[i]), std::make_pair(std::get<2>(layout[i]).first / total_value, std::get<2>(layout[i]).second)});
        return normalized_layout;
    };

    // Initialize the hash ring according to the chosen partitioning policy.
    DeviceLayout1D<std::pair<double,size_t>> layout_values;
    switch (config.partitioning_type) {
        case 0: // Round-Robin
        default: // Round-Robin is the default // TODO: Round-robin is currently the same as maximum utilization.
            get_property_layout(layout_values, cluster_properties, config, [](const ClusterProperties& prop, const ProcessingConfig& config, HostProperties& host, const int node_id, const int device_id) {
                return std::make_pair(static_cast<double>(1), get_available_memory(prop, config, node_id, device_id)); });
            hash_ring.init(normalizeDeviceLayout1D(layout_values), cluster_properties, config);
            break;
        case 1: // Maximum Utilization
            get_property_layout(layout_values, cluster_properties, config, [](const ClusterProperties& prop, const ProcessingConfig& config, HostProperties& host, const int node_id, const int device_id) {
                return std::make_pair(static_cast<double>(1), get_available_memory(prop, config, node_id, device_id)); });
            hash_ring.init(normalizeDeviceLayout1D(layout_values), cluster_properties, config);
            break;
        case 2: { // Proportional Maximum Utilization
            get_property_layout(layout_values, cluster_properties, config, [](const ClusterProperties& prop, const ProcessingConfig& config, HostProperties& host, const int node_id, const int device_id) {
                return std::make_pair(static_cast<double>(get_available_memory(prop, config, node_id, device_id)), get_available_memory(prop, config, node_id, device_id)); });
            // Initialize hash ring using normalized layout values.
            hash_ring.init(normalizeDeviceLayout1D(layout_values), cluster_properties, config);
            break; }
        case 3: // Newest Architecture First // TODO: Currently relative instead of arch-only. Change to send only to highest arch until full.
            get_property_layout(layout_values, cluster_properties, config, [](const ClusterProperties& prop, const ProcessingConfig& config, HostProperties& host, const int node_id, const int device_id) {
                return std::make_pair(static_cast<double>(prop.nodes[node_id].devices[device_id].compute_capability), get_available_memory(prop, config, node_id, device_id)); });
            // Initialize hash ring using normalized layout values.
            hash_ring.init(normalizeDeviceLayout1D(layout_values), cluster_properties, config);
            break;
        case 4: // Relative CUDA cores
            get_property_layout(layout_values, cluster_properties, config, [](const ClusterProperties& prop, const ProcessingConfig& config, HostProperties& host, const int node_id, const int device_id) {
                return std::make_pair(static_cast<double>(prop.nodes[node_id].devices[device_id].multiprocessor_count * prop.nodes[node_id].devices[device_id].cores_per_sm), get_available_memory(prop, config, node_id, device_id)); });
            // Initialize hash ring using normalized layout values.
            hash_ring.init(normalizeDeviceLayout1D(layout_values), cluster_properties, config);
            break;
        case 5: // Relative peak performance
            get_property_layout(layout_values, cluster_properties, config, [](const ClusterProperties& prop, const ProcessingConfig& config, HostProperties& host, const int node_id, const int device_id) {
                return std::make_pair(static_cast<double>(prop.nodes[node_id].devices[device_id].peak_performance), get_available_memory(prop, config, node_id, device_id)); });
            // Initialize hash ring using normalized layout values.
            hash_ring.init(normalizeDeviceLayout1D(layout_values), cluster_properties, config);
            break;
    }
}

/**
 * @brief Distribute the given query session to a device within the network.
 *
 * @param query_session The query session to distribute.
 * @param cluster_properties The properties of the devices within the network.
 * @param config The processing configuration.
 */
void distribute_query(const ClusterProperties& cluster_properties, const ProcessingConfig& config, HashRing& hash_ring, LocalPartitions& my_partitions, SERP_Hst& query_session) {
    // Get the destination node and device.
    std::pair<node, device> destination = hash_ring.get_destination(query_session.get_query());

    if (destination.first == config.node_id) { // If the destination is the current node, add the query session to the local partition.
        // Add the query session to the local partition.
        std::get<0>(my_partitions[destination.second]).push_back(query_session);
    }
    else { // If the destination is a remote node, send the query session to the destination.
        // Send the query session to the destination.
        Communicate::send_sessions(destination.first, destination.second, query_session); // Node, Device, Query Session
    }
}

/**
 * @brief Parses a raw dataset file into a vector where each query session is
 * grouped by its session id. The parsed vector is then distributed among the
 * devices within the network using the given partitioning scheme.
 *
 * @param dataset The dataset class which will contain the parsed results.
 * @param cluster_properties The properties of the devices within the network.
 * @param config The processing configuration.
 * @param hash_ring The hash ring containing the partitioning policy.
 *
 * @return Success of the parsing operation. (0 = success, 1 = failure)
 */
int process_dataset(LocalPartitions& my_partitions, const ClusterProperties& cluster_properties, const ProcessingConfig& config, HashRing& hash_ring) {
    // Open an input file stream using the given dataset path.
    std::ifstream raw_file(config.dataset_path);
    if (!raw_file.is_open()) {
        return 1;
    }

    std::string line;
    std::string element;
    SERP_Hst curr_SERP_Hst;
    int n_lines{0}, n_queries{0}, progress{0}, session_id_curr{0}, session_id_prev{-1};

    // Proceed to reading the dataset if the input stream has succesfully been opened.
    std::cout << "Raw dataset file \"" << config.dataset_path << "\" is opened." << std::endl;
    // Extract lines of type string from the dataset input stream, while the
    // number of extracted lines does not exceed the indicated maximum number of sessions.
    // The last session is not added to the dataset due to sessions only being
    // added when a new session id is encountered.
    while (std::getline(raw_file, line) && (n_queries < config.max_sessions || config.max_sessions == -1)) {
        // Create a stream for the line string and a string vector to store the
        // processed string.
        std::istringstream ssi(line);
        std::vector<std::string> line_vec;

        // Only add an element from the input stream to the current entire
        // line string if the element is errorless.
        while (ssi >> element) {
            line_vec.push_back(element);
        }

        try {
            session_id_curr = std::stoi(line_vec[0]);
        } catch (std::invalid_argument& e) {
            std::cout << "\nInvalid data: " << line << std::endl;
            continue;
        }

        n_lines++;

        // If this line contains 15 elements, then it describes a query action.
        // Vector elements: session id, time passed, Q, query id, region id,
        // document id (rank = 1), ..., document id (rank = MAX_SERP).
        if (line_vec.size() == QUERY_LINE_LENGTH) {
            // Add the previous query to the parsed dataset. Ensure that
            // this query session has actually been filled by checking
            // whether the session id does not equal its default value (-1).
            if (session_id_prev != -1) {
                distribute_query(cluster_properties, config, hash_ring, my_partitions, curr_SERP_Hst);
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
        int percent = static_cast<int>((static_cast<float>(n_queries) * 100.f) / config.max_sessions);
        if (percent != progress) {
            std::cout << "\r[" << std::string(percent / 5, '=') << std::string(20 - percent / 5, ' ') << "] " << percent << "%" << std::flush;
            progress = percent;
        }
    }

    // Communicate end of parsing.
    for (size_t nid = 1; nid < cluster_properties.node_count; nid++) {
        SERP_Hst end_of_parsing;
        Communicate::send_sessions(nid, 0, end_of_parsing); // Node, Device, Query Session
    }

    std::cout << "\rRead " << n_lines << " lines." << std::string(15, ' ') << std::endl;
    return 0;
}

/**
 * @brief Assign parameter indices to the query-document pairs in the train set.
 *
 * @param my_partition The train set.
 * @param thread_count The number of threads to use.
 * @param qd_map The query-document map.
 */
int assign_train_parameter_indices(TrainSet& my_partition, int thread_count, QueryDocumentMap& qd_map) {
    size_t param_index = 0; // Number of unique QD pairs in the training set.
    DMap::iterator doc_map;
    QDMap::iterator query_map;
    bool inserted;

    // Iterate over all query sessions in the training set.
    for (SERP_Hst& query : my_partition) {
        // Check if the query exists and insert it if it does not.
        std::tie(query_map, std::ignore) = qd_map.emplace(query.get_query(), DMap{});

        // Iterate over all documents in the query session.
        for (int did = 0; did < MAX_SERP; did++) {
            // Check if the query-document pair exists and insert it if it does not.
            std::tie(doc_map, inserted) = query_map->second.emplace(query[did].get_doc_id(), param_index);
            query.access_sr(did).set_param_index(doc_map->second);

            // If a new document was inserted, increment the parameter index.
            if (inserted) {
                param_index++;
            }
        }
    }

    return param_index;
}

/**
 * @brief Assign parameter indices to the query-document pairs in the test set
 * in parallel.
 *
 * @param my_partition The test set.
 * @param thread_count The number of threads to use.
 * @param qd_map The query-document map.
 */
void assign_test_parameter_indices(TestSet& my_partition, int thread_count, QueryDocumentMap& qd_map) {
    // Iterate over all query sessions in the training set.
    my_partition.erase(std::remove_if(my_partition.begin(), my_partition.end(),
        [&](SERP_Hst& query) {
            // Check if the query exists and remove from test set it if it does not.
            auto query_map = qd_map.find(query.get_query());
            if (query_map == qd_map.end()) {
                return true;  // Remove the query from the test set.
            }

            // Iterate over all documents in the query session.
            bool found_match = false;
            for (int did = 0; did < MAX_SERP; did++) {
                // Check if the query-document pair exists.
                auto doc_map = query_map->second.find(query[did].get_doc_id());
                if (doc_map != query_map->second.end()) {
                    found_match = true;
                    // Set parameter index of document in query equal to
                    // parameter index in train set stored.
                    query.access_sr(did).set_param_index(doc_map->second);
                }
            }
            // Remove the query from partition if no document exists.
            return !found_match;
        }), my_partition.end());
}


/**
 * @brief Split the partitions assigned to this node into training and testing
 * sets, group the partitions by query, and assign parameter indices to the
 * query-document pairs in the train and test sets.
 *
 * @param my_partitions The partitions assigned to this node.
 * @param config The processing configuration.
 */
void prepare_partitions(LocalPartitions& my_partitions, const ProcessingConfig& config) {
    // Split the partitions assigned to this node into training and testing sets.
    for (int pid = 0; pid < (config.exec_mode == 0 || config.exec_mode == 2 ? config.device_count : 1); pid++) {
        // Calculate the length of the train and test set using the test share.
        int n_queries = std::get<0>(my_partitions[pid]).size();
        int n_train = static_cast<int>(n_queries * (1 - config.test_share) - 0.5) + 1; // Round to nearest integer.
        int n_test = n_queries - n_train;
        // Move the tail of the training set to the test set.
        std::get<1>(my_partitions[pid]).assign(std::make_move_iterator(std::get<0>(my_partitions[pid]).begin() + n_train),
                                               std::make_move_iterator(std::get<0>(my_partitions[pid]).end()));
        // Clear the moved elemetns from the training set.
        std::get<0>(my_partitions[pid]).erase(std::get<0>(my_partitions[pid]).begin() + n_train, std::get<0>(my_partitions[pid]).end());
    }

    // Group query sessions by query id in both the training set by sorting the
    // query ids in ascending order.
    sort_partitions(my_partitions, config.thread_count);

    // Assign parameter indices to the query-document pairs in the train and
    // test sets.
    for (int pid = 0; pid < (config.exec_mode == 0 || config.exec_mode == 2 ? config.device_count : 1); pid++) {
        QueryDocumentMap qd_map;
        std::get<2>(my_partitions[pid]) = assign_train_parameter_indices(std::get<0>(my_partitions[pid]), config.thread_count, qd_map);
        assign_test_parameter_indices(std::get<1>(my_partitions[pid]), config.thread_count, qd_map);
    }
}

/**
 * @brief Parses a raw dataset file and distributes the parsed sessions among
 * the devices within the network using the given partitioning scheme.
 *
 * @param dataset The dataset class which will contain the parsed results.
 * @param cluster_properties The properties of the devices within the network.
 * @param config The processing configuration.
 * @return int Success of the parsing operation. (0 = success, 1 = failure)
 */
int parse_dataset(const ClusterProperties& cluster_properties, const ProcessingConfig& config, LocalPartitions& my_partitions) {
    // Reserve an estimate of the memory for partitions.
    for (int pid = 0; pid < (config.exec_mode == 0 || config.exec_mode == 2 ? config.device_count : 1); pid++) {
        std::get<0>(my_partitions[pid]).reserve(config.max_sessions * (1 - config.test_share) / config.device_count);
        std::get<1>(my_partitions[pid]).reserve(config.max_sessions * config.test_share / config.device_count);
    }

    // Parse the dataset and distribute the parsed sessions among the devices.
    if (config.node_id == ROOT_RANK) {
        // Initialize the hash ring with a partitioning policy.
        HashRing hash_ring;
        set_partitioning_policy(cluster_properties, config, hash_ring);
        // Parse the dataset and distribute the parsed sessions among the devices.
        if (process_dataset(my_partitions, cluster_properties, config, hash_ring))
            return 1;
    }
    else {
        // Receive the parsed sessions from the root node.
        while (Communicate::recv_sessions(ROOT_RANK, my_partitions) != -1) {};
    }
    Communicate::barrier(); // Wait for all nodes to finish parsing.

    // Prepare the partitions assigned to this node for use in the training.
    prepare_partitions(my_partitions, config);

    // Shrink the partitions assigned to this node to fit the actual number of
    // query sessions.
    for (int pid = 0; pid < (config.exec_mode == 0 || config.exec_mode == 2 ? config.device_count : 1); pid++) {
        std::get<0>(my_partitions[pid]).shrink_to_fit();
        std::get<1>(my_partitions[pid]).shrink_to_fit();
    }
    Communicate::barrier(); // Wait for all nodes to finish preparing.

    return 0;
}
