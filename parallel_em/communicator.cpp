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
     */
    void initiate(int& argc, char**& argv, int& n_nodes, int& node_id) {
        // Initialize MPI state.
        MPI_CHECK(MPI_Init(&argc, &argv));

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
     * @brief Communicate the number of devices to the root node.
     */
    void get_n_devices(const int& n_devices, int* n_devices_network) {
        MPI_CHECK(MPI_Gather(&n_devices, 1, MPI_INT, n_devices_network, 1, MPI_INT, ROOT, MPI_COMM_WORLD));
    }

    /**
     * @brief Gather the compute architectures and free memory on the root node.
     */
    void gather_properties(const int& node_id, const int& n_nodes, const int& n_devices, int* n_devices_network, std::vector<std::vector<std::vector<int>>>& network_properties, const int* device_architecture, const int* free_memory) {
        if (node_id == ROOT) {
            // Calculate the displacements of each node's device information. Use
            // this to allow variable length received messages.
            int displacements[n_nodes], cumm = 0, t_devices = Utils::sum(n_devices_network, n_nodes);
            for (int nid = 0; nid < n_nodes; nid++) {
                displacements[nid] = cumm;
                cumm += n_devices_network[nid];
            }

            // Gather the device information from all nodes in the network.
            int darch_network[t_devices], fmem_network[t_devices];
            MPI_CHECK(MPI_Gatherv(device_architecture, n_devices, MPI_INT, darch_network, n_devices_network, displacements, MPI_INT, ROOT, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gatherv(free_memory, n_devices, MPI_INT, fmem_network, n_devices_network, displacements, MPI_INT, ROOT, MPI_COMM_WORLD));

            // Keep track of the network information using the network properties array.
            int ibuf = 0;
            for (int nid = 0; nid < network_properties.size(); nid++) {
                std::vector<std::vector<int>> node_devices(n_devices_network[nid]);
                for (int did = 0; did < n_devices_network[nid]; did++, ibuf++) {
                    node_devices[did].insert(node_devices[did].end(), { darch_network[ibuf], fmem_network[ibuf] });
                }
                network_properties[nid] = node_devices;
            }
        }
        else {
            // Send the device architecture and free memory to the root node.
            MPI_CHECK(MPI_Gatherv(device_architecture, n_devices, MPI_INT, NULL, NULL, NULL, MPI_INT, ROOT, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gatherv(free_memory, n_devices, MPI_INT, NULL, NULL, NULL, MPI_INT, ROOT, MPI_COMM_WORLD));
        }
    }

    /**
     * @brief Communicate the training sets for each device to their node.
     */
    void send_partitions(const int& node_id, const int& n_nodes, const int& n_devices, const int& total_n_devices, const int* n_devices_network, Dataset& dataset, std::vector<std::tuple<std::vector<SERP_HST>, std::vector<SERP_HST>, int>>& device_partitions, std::vector<std::unordered_map<int, std::unordered_map<int, int>>*>& root_mapping) {
        // Create SERP_HST MPI datatype.
        MPI_Datatype MPI_SERP;
        MPI_CHECK(MPI_Type_contiguous(sizeof(SERP_HST) / sizeof(int), MPI_INT, &MPI_SERP));
        MPI_CHECK(MPI_Type_commit(&MPI_SERP));

        // Communicate the training sets for each device to their node.
        if (node_id == ROOT) {
            std::cout << "\nCommunicating " << total_n_devices << " partitions to " << n_nodes << " machines." << std::endl;
            MPI_Request request = MPI_REQUEST_NULL;

            // Send the partitions to the other nodes.
            for (int nid = 0; nid < n_nodes; nid++) {
                for (int did = 0; did < n_devices_network[nid]; did++) {
                    // Exclude the root node from sending targets.
                    if (nid != ROOT) {
                        // Send the train set to the node.
                        MPI_CHECK(MPI_Isend(dataset.get_train_set(nid, did)->data(), dataset.get_train_set(nid, did)->size(), MPI_SERP, nid, dataset.size_qd(nid, did), MPI_COMM_WORLD, &request));
                        MPI_CHECK(MPI_Isend(dataset.get_test_set(nid, did)->data(), dataset.get_test_set(nid, did)->size(), MPI_SERP, nid, dataset.size_qd(nid, did), MPI_COMM_WORLD, &request));
                    }
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
                std::get<2>(device_partitions[did]) = dataset.size_qd(node_id, did);

                root_mapping[did] = dataset.get_mapping(node_id, did);
            }
        }
        else {
            // Receive the train and test sets from the root node.
            for (int device_id = 0; device_id < n_devices; device_id++) {

                // Receive twice for every device. Both the train and test sets.
                for (int i = 0; i < 2; i++) {
                    // Probe for the size of the message.
                    MPI_Status status;
                    MPI_CHECK(MPI_Probe(ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
                    int msg_length;
                    MPI_CHECK(MPI_Get_count(&status, MPI_SERP, &msg_length));

                    // Select the train or test partition depending on the received tag.
                    std::vector<SERP_HST>* partition_ptr;
                    if (i == 0) {
                        partition_ptr = &(std::get<0>(device_partitions[device_id]));
                        std::get<2>(device_partitions[device_id]) = status.MPI_TAG;
                    }
                    else if (i == 1) {
                        partition_ptr = &(std::get<1>(device_partitions[device_id]));
                    }

                    // Allocate memory for the set.
                    partition_ptr->resize(msg_length);
                    // Receive the set.
                    MPI_CHECK(MPI_Recv(partition_ptr->data(), msg_length, MPI_SERP, ROOT, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
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
    void exchange_parameters(std::vector<std::vector<std::vector<Param>>>& dest, const std::vector<std::vector<Param>>& my_params, const int n_nodes, const int node_id) {
        // Create Param MPI datatype.
        MPI_Datatype MPI_PARAM;
        MPI_CHECK(MPI_Type_contiguous(sizeof(Param) / sizeof(float), MPI_FLOAT, &MPI_PARAM));
        MPI_CHECK(MPI_Type_commit(&MPI_PARAM));

        // Send this node's synchronized public device parameters to all other
        std::vector<Param> receive_buffer; // Parameter type -> Parameters.
        for (int param_type = 0; param_type < my_params.size(); param_type++) {
            receive_buffer.resize(n_nodes * my_params[param_type].size());

            // Gather the results from all other nodes.
            MPI_CHECK(MPI_Allgather(my_params[param_type].data(),
                                    my_params[param_type].size(),
                                    MPI_PARAM,
                                    receive_buffer.data(),
                                    my_params[param_type].size(),
                                    MPI_PARAM, MPI_COMM_WORLD));

            // Sort the results by node.
            int nid{0}, param_index{0};
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
    void gather_evaluations(std::map<int, std::array<float, 2>>& loglikelihood, std::map<int, Perplexity>& perplexity, const int n_nodes, const int node_id, const int* n_devices_network) {
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
        if (node_id == ROOT) {
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
            MPI_CHECK(MPI_Gatherv(tmp_llh_vals.data(), tmp_llh_vals.size(), MPI_LLH, llh_vals, n_devices_network, displacements, MPI_LLH, ROOT, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gatherv(tmp_ppl_vals.data(), tmp_ppl_vals.size(), MPI_PPL, ppl_vals, n_devices_network, displacements, MPI_PPL, ROOT, MPI_COMM_WORLD));

            // Deserialize the received results on the root node.
            int did{0};
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
        else {
            // Send the device architecture and free memory to the root node.
            MPI_CHECK(MPI_Gatherv(tmp_llh_vals.data(), tmp_llh_vals.size(), MPI_LLH, NULL, NULL, NULL, MPI_LLH, ROOT, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Gatherv(tmp_ppl_vals.data(), tmp_ppl_vals.size(), MPI_PPL, NULL, NULL, NULL, MPI_PPL, ROOT, MPI_COMM_WORLD));
        }
    }
}