/** MPI communication functions.
 *
 * communicate.cpp:
 *  - Defines several MPI functions used to communicate across different machines.
 */

#include "communicator.h"

namespace Communicate
{
    /**
     * @brief Initiates MPI state. Sets the number of nodes and this node's ID.
     *
     * @param argc Number of arguments passed to the program.
     * @param argv Arguments passed to the program.
     * @param n_nodes The number of nodes in the network.
     * @param node_id The ID of the current node.
     */
    void initiate(int& argc, char**& argv, int& n_nodes, int& node_id) {
        // Initialize MPI state and request multi-threading support. Indicate
        // that only the master thread will make MPI calls (FUNNELED).
        int provided;
        MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        if (node_id == ROOT_RANK && provided < MPI_THREAD_FUNNELED) {
            std::cerr << "\033[12;33mWarning\033[0m: MPI did not provide requested level of multithreading support. "
                      << "Multithreaded execution will be serialized. "
                      << "Has MPI been configured for multithreading usage?"
                      << std::endl;
        }

        // Get our MPI node number and node count.
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_nodes));
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
    }

    /**
     * @brief End the MPI state.
     */
    void finalize(void) {
        MPI_Finalize();
    }

    /**
     * @brief Check if any node has raised a non-critical error. If so, print
     * the error on the failed node and have all nodes exit.
     *
     * @param err_msg The error message if there is any.
     */
    void error_check(std::string err_msg /* = "" */) {
        int error = !err_msg.empty() ? 1 : 0;
        if (!err_msg.empty()) {
            std::cerr << err_msg << std::endl;
        }

        // Use MPI_Allreduce to check if any node has send a non-zero error code.
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

        // Exit if any node raised an error.
        if (error) {
            finalize();
            exit(EXIT_SUCCESS);
        }
    }

    /**
     * @brief Communicate the number of devices to the root node.
     *
     * @param n_devices The number of devices on the current node.
     * @param n_devices_network The number of devices on each node.
     */
    void get_n_devices(const int& n_devices, int* n_devices_network) {
        MPI_CHECK(MPI_Gather(&n_devices, 1, MPI_INT, n_devices_network, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD)); // Sender & Receiver
    }

    /**
     * @brief Gather the compute architectures and free memory on the root node.
     *
     * @param node_id The ID of the current node.
     * @param n_nodes The number of nodes in the network.
     * @param n_devices The number of devices on the current node.
     * @param n_devices_network The number of devices on each node.
     * @param network_properties The number of devices per node.
     * @param device_architectures The compute architectures of the devices on
     * each node.
     * @param free_memory The free memory on each device on each node.
     */
    void gather_properties(const int& n_devices, int* n_devices_network, std::vector<std::vector<std::vector<int>>>& network_properties, const int* device_architecture, const int* free_memory) {
        // Get the number of nodes and the current node rank.
        int node_id, n_nodes;
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_nodes));

        if (node_id == ROOT_RANK) { // Receiver
            // Calculate the displacements of each node's device information. Use
            // this to allow variable length received messages.
            int displacements[n_nodes], cumm = 0, t_devices = std::accumulate(n_devices_network, n_devices_network+n_nodes, 0);
            for (int nid = 0; nid < n_nodes; nid++) {
                displacements[nid] = cumm;
                cumm += n_devices_network[nid];
            }

            // Gather the device information from all nodes in the network.
            int darch_network[t_devices], fmem_network[t_devices];
            MPI_CHECK(MPI_Gatherv(device_architecture, n_devices, MPI_INT, darch_network, n_devices_network, displacements, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gatherv(free_memory, n_devices, MPI_INT, fmem_network, n_devices_network, displacements, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));

            // Keep track of the network information using the network properties array.
            int ibuf = 0;
            for (size_t nid = 0; nid < network_properties.size(); nid++) {
                std::vector<std::vector<int>> node_devices(n_devices_network[nid]);
                for (int did = 0; did < n_devices_network[nid]; did++, ibuf++) {
                    node_devices[did].insert(node_devices[did].end(), { darch_network[ibuf], fmem_network[ibuf] });
                }
                network_properties[nid] = node_devices;
            }
        }
        else { // Sender
            // Send the device architecture and free memory to the root node.
            MPI_CHECK(MPI_Gatherv(device_architecture, n_devices, MPI_INT, NULL, NULL, NULL, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gatherv(free_memory, n_devices, MPI_INT, NULL, NULL, NULL, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
        }
    }

    /**
     * @brief Gather the properties of all nodes in the cluster.
     *
     * @param local_node_properties The properties of the local node.
     * @return ClusterProperties The properties of all nodes in the cluster.
     */
    ClusterProperties gather_properties_all(NodeProperties local_node_properties) {
        // Get the number of nodes and the current node rank.
        int node_id, n_nodes;
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_nodes));

        // Communicate the number of devices on each node to all nodes.
        int devices_per_node[n_nodes];
        MPI_CHECK(MPI_Allgather(&local_node_properties.host.device_count, 1, MPI_INT, devices_per_node, 1, MPI_INT, MPI_COMM_WORLD));

        // Create an MPI datatype for the HostProperties struct.
        std::vector<HostProperties> host_properties(n_nodes);
        int n_members_hst = 6;
        int block_lengths_hst[6] = {1, 1, 1, 1, 1, HOST_NAME_MAX};
        MPI_Datatype types_hst[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CHAR};
        MPI_Datatype MPI_HST_PROP;
        MPI_Aint offsets_hst[6];

        offsets_hst[0] = offsetof(HostProperties, node_id);
        offsets_hst[1] = offsetof(HostProperties, device_count);
        offsets_hst[2] = offsetof(HostProperties, thread_count);
        offsets_hst[3] = offsetof(HostProperties, free_memory);
        offsets_hst[4] = offsetof(HostProperties, total_memory);
        offsets_hst[5] = offsetof(HostProperties, host_name);

        MPI_Type_create_struct(n_members_hst, block_lengths_hst, offsets_hst, types_hst, &MPI_HST_PROP);
        MPI_Type_commit(&MPI_HST_PROP);

        // Communicate the host properties to all nodes.
        MPI_CHECK(MPI_Allgather(&local_node_properties.host, 1, MPI_HST_PROP, host_properties.data(), 1, MPI_HST_PROP, MPI_COMM_WORLD));

        // Create an MPI datatype for the DeviceProperties struct.
        std::vector<DeviceProperties> device_properties(
            std::accumulate(devices_per_node, devices_per_node + n_nodes, 0));
        int n_members_dev = 17;
        int block_lengths_dev[17] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 256};
        MPI_Datatype types_dev[17] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CHAR};
        MPI_Datatype MPI_DEV_PROP;
        MPI_Aint offsets_dev[17];

        offsets_dev[0] = offsetof(DeviceProperties, device_id);
        offsets_dev[1] = offsetof(DeviceProperties, compute_capability);
        offsets_dev[2] = offsetof(DeviceProperties, registers_per_block);
        offsets_dev[3] = offsetof(DeviceProperties, registers_per_sm);
        offsets_dev[4] = offsetof(DeviceProperties, threads_per_block);
        offsets_dev[5] = offsetof(DeviceProperties, threads_per_sm);
        offsets_dev[6] = offsetof(DeviceProperties, warp_size);
        offsets_dev[7] = offsetof(DeviceProperties, memory_clock_rate);
        offsets_dev[8] = offsetof(DeviceProperties, memory_bus_width);
        offsets_dev[9] = offsetof(DeviceProperties, cores_per_sm);
        offsets_dev[10] = offsetof(DeviceProperties, clock_rate);
        offsets_dev[11] = offsetof(DeviceProperties, multiprocessor_count);
        offsets_dev[12] = offsetof(DeviceProperties, total_global_memory);
        offsets_dev[13] = offsetof(DeviceProperties, shared_memory_per_block);
        offsets_dev[14] = offsetof(DeviceProperties, total_constant_memory);
        offsets_dev[15] = offsetof(DeviceProperties, peak_performance);
        offsets_dev[16] = offsetof(DeviceProperties, device_name);

        MPI_Type_create_struct(n_members_dev, block_lengths_dev, offsets_dev, types_dev, &MPI_DEV_PROP);
        MPI_Type_commit(&MPI_DEV_PROP);

        // Calculate displacement necessary to receive varying numbers of devices.
        int displacements[n_nodes];
        displacements[0] = 0; // The first displacement is always 0.
        for (int i = 1; i < n_nodes; i++) { // Calculate the displacements for the remaining nodes.
            displacements[i] = displacements[i-1] + devices_per_node[i-1];
        }

        // Communicate the device properties to all nodes.
        MPI_CHECK(MPI_Allgatherv(local_node_properties.devices.data(), local_node_properties.devices.size(), MPI_DEV_PROP, device_properties.data(), devices_per_node, displacements, MPI_DEV_PROP, MPI_COMM_WORLD));

        // Populate cluster properties struct.
        ClusterProperties cluster_properties;
        cluster_properties.node_count = n_nodes;
        cluster_properties.device_count = std::accumulate(devices_per_node, devices_per_node + n_nodes, 0);
        cluster_properties.nodes = std::vector<NodeProperties>(n_nodes);

        // Populate node properties struct.
        for (int nid = 0; nid < n_nodes; nid++) {
            cluster_properties.nodes[nid].host = host_properties[nid];
            cluster_properties.nodes[nid].devices = std::vector<DeviceProperties>(devices_per_node[nid]);
            for (int did = 0; did < devices_per_node[nid]; did++) {
                cluster_properties.nodes[nid].devices[did] = device_properties[displacements[nid] + did];
            }
        }

        // Free the MPI datatypes.
        MPI_Type_free(&MPI_HST_PROP);
        MPI_Type_free(&MPI_DEV_PROP);

        return cluster_properties;
    }

    /**
     * @brief Communicate the training sets for each device to their node.
     *
     * @param node_id The ID of the current node.
     * @param n_nodes The number of nodes in the network.
     * @param n_devices The number of devices on the current node.
     * @param n_devices_network The number of devices on each node.
     * @param dataset The source training set to be distributed.
     * @param device_partitions The destination training set partitions for
     * each device.
     */
    void send_partitions(const int& n_devices, const int& total_n_devices, const int* n_devices_network, Dataset& dataset, LocalPartitions& device_partitions) {
        // Get the number of nodes and the current node rank.
        int node_id, n_nodes;
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_nodes));

        // Create SERP_Hst MPI datatype.
        MPI_Datatype MPI_SERP;
        MPI_CHECK(MPI_Type_contiguous(sizeof(SERP_Hst) / sizeof(int), MPI_INT, &MPI_SERP));
        MPI_CHECK(MPI_Type_commit(&MPI_SERP));

        // Communicate the training sets for each device to their node.
        if (node_id == ROOT_RANK) { // Sender
            std::cout << "\nCommunicating " << total_n_devices << " partitions to " << n_nodes << " machines." << std::endl;
            MPI_Request request = MPI_REQUEST_NULL;

            // Send the partitions to the other nodes excluding the root node.
            for (int nid = 1; nid < n_nodes; nid++) {
                for (int did = 0; did < n_devices_network[nid]; did++) {
                    // Send the train set to the node.
                    int size_qd = dataset.get_query_doc_pair_size(nid, did);
                    MPI_CHECK(MPI_Isend(dataset.get_train_set(nid, did)->data(), (int) dataset.get_train_set(nid, did)->size(), MPI_SERP, nid, 0, MPI_COMM_WORLD, &request));
                    MPI_CHECK(MPI_Isend(dataset.get_test_set(nid, did)->data(), (int) dataset.get_test_set(nid, did)->size(), MPI_SERP, nid, 0, MPI_COMM_WORLD, &request));
                    MPI_CHECK(MPI_Isend(&size_qd, 1, MPI_INT, nid, 0, MPI_COMM_WORLD, &request));
                }
            }
            MPI_CHECK(MPI_Wait(&request, MPI_STATUS_IGNORE));

            // Also assign partitions to this device. Move the train and test sets,
            // instead of copying them to the device partitions, to save memory.
            for (int did = 0; did < n_devices; did++) {
                std::get<0>(device_partitions[did]) = std::move(*dataset.get_train_set(node_id, did));
                (*dataset.get_train_set(node_id, did)).erase(std::begin(*dataset.get_train_set(node_id, did)), std::end(*dataset.get_train_set(node_id, did)));
                std::get<1>(device_partitions[did]) = std::move(*dataset.get_test_set(node_id, did));
                (*dataset.get_test_set(node_id, did)).erase(std::begin(*dataset.get_test_set(node_id, did)), std::end(*dataset.get_test_set(node_id, did)));
                std::get<2>(device_partitions[did]) = dataset.get_query_doc_pair_size(node_id, did);
            }
        }
        else { // Receiver
            // Receive the train and test sets from the root node.
            for (int did = 0; did < n_devices; did++) {
                for (int msg_type = 0; msg_type < 3; msg_type++) {
                    // Select the train or test partition according to message ordering (uses FIFO).
                    if (msg_type == 0 || msg_type == 1) {
                        UnassignedSet* dataset_ptr;

                        // Probe for the size of the message.
                        MPI_Status status;
                        MPI_CHECK(MPI_Probe(ROOT_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
                        int msg_length;
                        MPI_CHECK(MPI_Get_count(&status, MPI_SERP, &msg_length));

                        if (msg_type == 0) { // Train set.
                            dataset_ptr = &(std::get<0>(device_partitions[did]));
                        }
                        else { // Test set.
                            dataset_ptr = &(std::get<1>(device_partitions[did]));
                        }

                        // Allocate memory for the set.
                        dataset_ptr->resize(msg_length);
                        // Receive the set.
                        MPI_CHECK(MPI_Recv(dataset_ptr->data(), msg_length, MPI_SERP, ROOT_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    }
                    else if (msg_type == 2) { // Number of the query-document parameters.
                        // Receive the parameter size.
                        MPI_CHECK(MPI_Recv(&std::get<2>(device_partitions[did]), 1, MPI_INT, ROOT_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    }
                }
            }
        }
    }

    /**
     * @brief Blocks execution until all processes have reached this function.
     */
    void barrier(void) {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * @brief Exchanges sets of parameters between all nodes in the network.
     *
     * @param dest A multi-dimensional vector storing the parameters of each node
     * in the network. The vector is structured as follows: Node ID -> Parameter
     * type -> Parameters.
     * @param my_params The parameters which this node will share with the other
     * nodes in the network. The vector is structured as follows: Parameter type ->
     * Parameters.
     * @param n_nodes The number of nodes in the network.
     * @param node_id The MPI communication rank of this node.
     */
    void exchange_parameters(std::vector<std::vector<std::vector<Param>>>& dest, const std::vector<std::vector<Param>>& my_params) {
        // Get the number of nodes.
        int n_nodes;
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_nodes));

        // Create Param MPI datatype.
        MPI_Datatype MPI_PARAM;
        MPI_CHECK(MPI_Type_contiguous(sizeof(Param) / sizeof(float), MPI_FLOAT, &MPI_PARAM));
        MPI_CHECK(MPI_Type_commit(&MPI_PARAM));

        // Send this node's synchronized public device parameters to all other
        std::vector<Param> receive_buffer; // Parameter type -> Parameters.
        for (size_t param_type = 0; param_type < my_params.size(); param_type++) { // Sender & Receiver
            receive_buffer.resize(n_nodes * my_params[param_type].size());

            // Gather the results from all other nodes.
            MPI_CHECK(MPI_Allgather(my_params[param_type].data(),
                                    my_params[param_type].size(),
                                    MPI_PARAM,
                                    receive_buffer.data(),
                                    my_params[param_type].size(),
                                    MPI_PARAM, MPI_COMM_WORLD));

            // Sort the results by node.
            size_t nid{0}, param_index{0};
            for (Param param : receive_buffer) {
                if (param_index == my_params[param_type].size()) {
                    param_index = 0;
                    nid++;
                }

                dest[nid][param_type][param_index] = param;
                param_index++;
            }
        }
    }

    /**
     * @brief Compute the result of combining the PBM parameters from other nodes
     * or devices.
     *
     * @param parameters A multi-dimensional vector containing the parameters to be
     * combined. The vector is structured as follows: Node/Device ID -> Parameter
     * type -> Parameters.
     */
    void sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters) {
        for (size_t rank = 0; rank < parameters[0][0].size(); rank++) {
            for (size_t param_type = 0; param_type < parameters[0].size(); param_type++) {
                Param base = parameters[0][param_type][rank];
                // Subtract the starting values of other datasets.
                parameters[0][param_type][rank].set_values(base.numerator_val() - (parameters.size() - 1),
                                                        base.denominator_val() - 2 * (parameters.size() - 1));

                for (size_t device_id = 1; device_id < parameters.size(); device_id++) {
                    Param ex = parameters[device_id][param_type][rank];
                    parameters[0][param_type][rank].add_to_values(ex.numerator_val(),
                                                                ex.denominator_val());
                }
            }
        }
    }

    /**
     * @brief Gathers the evaluation results from all nodes within the network on
     * the root node.
     *
     * @param loglikelihood The map containing the log-likelihood calculated for
     * each device's partition on this node.
     * @param perplexity The map containing the perplexity calculated for each
     * device's partition on this node.
     * @param n_nodes The number of nodes in the network.
     * @param node_id The MPI communication rank of this node.
     * @param n_devices_network The number of devices per node in the network.
     */
    void gather_evaluations(std::map<int, std::array<float, 2>>& loglikelihood, std::map<int, Perplexity>& perplexity, const int* n_devices_network) {
        // Get the number of nodes and the current node rank.
        int node_id, n_nodes;
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_nodes));

        // Create log-likelihood MPI datatype.
        MPI_Datatype MPI_LLH;
        MPI_CHECK(MPI_Type_contiguous(sizeof(std::array<float, 2>) / sizeof(float), MPI_FLOAT, &MPI_LLH));
        MPI_CHECK(MPI_Type_commit(&MPI_LLH));

        // Create perplexity MPI datatype.
        MPI_Datatype MPI_PPL;
        MPI_CHECK(MPI_Type_contiguous(sizeof(Perplexity) / sizeof(float), MPI_FLOAT, &MPI_PPL));
        MPI_CHECK(MPI_Type_commit(&MPI_PPL));

        // Temporarily serialize the log-likelihood and perplexity maps.
        std::vector<std::array<float, 2>> tmp_llh_vals;
        std::vector<std::array<float, sizeof(Perplexity) / sizeof(float)>> tmp_ppl_vals;

        std::map<int, std::array<float, 2>>::iterator llh_itr = std::begin(loglikelihood);
        for (; llh_itr != std::end(loglikelihood); llh_itr++) {
            tmp_llh_vals.push_back(llh_itr->second);
        }

        std::map<int, Perplexity>::iterator ppl_itr = std::begin(perplexity);
        for (int did = 0; ppl_itr != std::end(perplexity); ppl_itr++, did++) {
            tmp_ppl_vals.resize(did + 1);
            std::copy(std::begin(ppl_itr->second.task_rank_perplexities), std::end(ppl_itr->second.task_rank_perplexities), std::begin(tmp_ppl_vals[did]));
            tmp_ppl_vals[did][ppl_itr->second.task_rank_perplexities.size()] = ppl_itr->second.task_size;
        }

        // Gather the log-likelihood and perplexity on the root node.
        if (node_id == ROOT_RANK) { // Receiver
            // Calculate the displacements of each node's evaluations.
            int t_devices{0};
            for (int nid = 0; nid < n_nodes; nid++) { t_devices += n_devices_network[nid]; }
            int displacements[n_nodes], cumm = 0;
            for (int nid = 0; nid < n_nodes; nid++) {
                displacements[nid] = cumm;
                cumm += n_devices_network[nid];
            }

            // Create the receive buffers.
            std::array<float, 2> llh_vals[t_devices];
            std::array<float, sizeof(Perplexity) / sizeof(float)> ppl_vals[t_devices];

            // Gather the device information from all nodes in the network.
            MPI_CHECK(MPI_Gatherv(tmp_llh_vals.data(), tmp_llh_vals.size(), MPI_LLH, llh_vals, n_devices_network, displacements, MPI_LLH, ROOT_RANK, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gatherv(tmp_ppl_vals.data(), tmp_ppl_vals.size(), MPI_PPL, ppl_vals, n_devices_network, displacements, MPI_PPL, ROOT_RANK, MPI_COMM_WORLD));

            // Deserialize the received results on the root node.
            for (int did = 0; did < t_devices; did++) {
                // Insert the received log-likelihood values.
                loglikelihood[did] = llh_vals[did];

                // Insert the received perplexity values.
                Perplexity new_ppl;
                float task_size = ppl_vals[did][sizeof(Perplexity) / sizeof(float) - 1];
                std::array<float, 10UL> task_rank_perplexities;
                std::copy(std::begin(ppl_vals[did]), &ppl_vals[did][sizeof(Perplexity) / sizeof(float) - 1], std::begin(task_rank_perplexities));
                new_ppl.import(task_rank_perplexities, task_size);
                perplexity[did] = new_ppl;
            }
        }
        else { // Sender
            // Send the device architecture and free memory to the root node.
            MPI_CHECK(MPI_Gatherv(tmp_llh_vals.data(), tmp_llh_vals.size(), MPI_LLH, NULL, NULL, NULL, MPI_LLH, ROOT_RANK, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gatherv(tmp_ppl_vals.data(), tmp_ppl_vals.size(), MPI_PPL, NULL, NULL, NULL, MPI_PPL, ROOT_RANK, MPI_COMM_WORLD));
        }
    }

    /**
     * @brief Output the parameters to a file.
     *
     * @param node_id The ID of the current node.
     * @param processing_units The number of processing units on the current node.
     * @param file_path The base path to the file to write the parameters to.
     * @param dataset_partitions The dataset partitions for each processing unit.
     * @param headers The names of the parameters.
     * @param parameters The parameters to write to file.
     */
    void output_parameters(const int processing_units, const std::string file_path,
        const LocalPartitions& dataset_partitions,
        const std::pair<std::vector<std::string>, std::vector<std::string>> &headers,
        const std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> *parameters) {
        // Get the current node rank.
        int node_id;
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &node_id));

        // Write parameters shared by all nodes using only the root node.
        if (node_id == ROOT_RANK) {
            // Write public parameters to one or more files.
            for (size_t public_param_num = 0; public_param_num < parameters[0].first.size(); public_param_num++) {
                std::ofstream file; // Open file for current public parameter.
                std::cout << "  Writing parameter " << headers.first[public_param_num] << " to " << file_path + "_" + headers.first[public_param_num] + ".csv" << std::endl;
                file.open(file_path + "_" + headers.first[public_param_num] + ".csv");
                file << "rank, " << headers.first[public_param_num] << std::endl; // Write header.
                for (size_t rank = 0; rank < (*parameters[0].first[public_param_num]).size(); rank++) { // Write public parameters.
                    file << rank << ", " << (*parameters[0].first[public_param_num])[rank].value() << std::endl;
                }
                file.close();
            }

            // Write private parameter headers to one or more files.
            for (size_t private_param_num = 0; private_param_num < parameters[0].second.size(); private_param_num++) {
                std::ofstream file; // Open file for current private parameter.
                file.open(file_path + "_" + headers.second[private_param_num] + ".csv");
                std::cout << "  Writing parameter " << headers.second[private_param_num] << " to " << file_path + "_" + headers.second[private_param_num] + ".csv" << std::endl;
                file << "query, document, " << headers.second[private_param_num] << std::endl; // Write header.
                file.close();
            }
        }

        // Write private parameters to one or more files from all nodes.
        for (size_t private_param_num = 0; private_param_num < parameters[0].second.size(); private_param_num++) {
            // Retrieve highest number of required MPI_File_write_at_all calls using allreduce.
            int current_write_call = 0;
            int max_write_calls = processing_units;
            MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &max_write_calls, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

            MPI_File file;
            MPI_Status status;
            MPI_Offset msg_offset;

            // Open the file.
            MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, (file_path + "_" + headers.second[private_param_num] + ".csv").c_str(),
                                    MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file));

            // Iterate over all devices on the current node.
            for (size_t did = 0; did < processing_units; did++) {
                // Retrieve pointer to required dataset partition
                std::string data = "";

                for (size_t i = 0; i < std::get<0>(dataset_partitions[did]).size(); i++) {
                    SERP_Hst query_session = std::get<0>(dataset_partitions[did])[i];
                    int query = query_session.get_query();

                    for (size_t rank = 0; rank < MAX_SERP; rank++) {
                        int document = query_session[rank].get_doc_id();
                        SearchResult_Hst sr = query_session[rank];

                        float def{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
                        float val = sr.get_param_index() != -1 ? (*parameters[did].second[private_param_num])[sr.get_param_index()].value() : def;
                        data += std::to_string(query) + ", " + std::to_string(document) + ", " + std::to_string(val) + "\n";
                    }
                }

                // Compute the offset for each node.
                int msg_size = data.size();
                MPI_Exscan(&msg_size, &msg_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);

                // The first process does not participate in the exclusive scan, so it starts writing at the beginning of the file.
                if (node_id == 0) msg_offset = 0;

                // Write to the file (ordering is not preserved)
                MPI_CHECK(MPI_File_write_at_all(file, msg_offset, data.data(), data.size(), MPI_CHAR, &status));
                current_write_call++;
            }

            // Perform dummy writes to ensure all nodes have written the same number of times.
            while (current_write_call < max_write_calls) {
                // Compute the offset for each node
                int msg_size = 0;
                MPI_Exscan(&msg_size, &msg_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
                // The first process does not participate in the exclusive scan, so it should start writing at the beginning of the file
                if (node_id == 0) msg_offset = 0;
                // Write dummy text to the file
                MPI_CHECK(MPI_File_write_at_all(file, msg_offset, "", 0, MPI_CHAR, &status));
                current_write_call++;
            }

            // Close the file
            MPI_CHECK(MPI_File_close(&file));
        }
    }
}