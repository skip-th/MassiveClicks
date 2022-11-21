/** Parallelizing EM on GPU(s).
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * parallel_em.cu:
 *  - Defines the functions used for initiating the EM process on the GPU.
 */

// User include.
#include "parallel_em.cuh"


//---------------------------------------------------------------------------//
// Host-side computation.                                                    //
//---------------------------------------------------------------------------//

/**
 * @brief Runs the Expectation-Maximization algorithm for a given click model
 * on one or multiple GPU's and evaluates the result.
 *
 * @param model_type The type of click model (e.g. 0 = PBM).
 * @param node_id The MPI communication rank of this node.
 * @param n_nodes The number of nodes in the network.
 * @param n_threads The number of CPU threads on this node.
 * @param n_devices_network The number of devices per node in the network.
 * @param n_itr The number of iterations for which the EM algorithm should run.
 * @param exec_mode The mode of execution (e.g. 0 = CPU, 1 = GPU).
 * @param n_devices The number of GPU devices on this node.
 * @param processing_units The number of compute devices on this node (incuding
 * CPU depending on the execution mode).
 * @param device_partitions The training and testing sets and the number of
 * query document pairs in the training set, for each device on this node.
 * @param output_path The path to the output file.
 */
void em_parallel(const int model_type, const int node_id, const int n_nodes,
    const int n_threads, const int* n_devices_network, const int n_itr,
    const int exec_mode, const int n_devices, const int processing_units,
    std::vector<std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>>& device_partitions,
    std::string output_path) {

    if (node_id == ROOT) {
        std::cout << "\nExpectation Maximization (EM) in parallel ..." << std::endl;
    }


    //-----------------------------------------------------------------------//
    // Initate host-side click model.                                        //
    //-----------------------------------------------------------------------//

    // Initiate a host-side click model for each device or for the host.
    ClickModel_Hst* cm_hosts[processing_units];
    for (int unit = 0; unit < processing_units; unit++) {
        // Initialize the click model.
        cm_hosts[unit] = create_cm_host(model_type);
        // Print a confirmation message on the first device of the root node.
        if (node_id == ROOT && unit == 0) {
            cm_hosts[unit]->say_hello();
        }
    }


    //-----------------------------------------------------------------------//
    // Assign queries to CPU threads.                                        //
    //-----------------------------------------------------------------------//

    std::vector<std::vector<int>> thread_start_idx(processing_units);
    if (exec_mode != 0) {
        int n_queries_total = std::accumulate(device_partitions.begin(), device_partitions.end(), 0,
            [](int sum, const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition) {
            return sum + std::get<0>(partition).size();});
        int available_threads = n_threads;
        int device_id = 0;
        // Allocate CPU threads to each device based on the number of assigned
        // queries.
        while (available_threads > 0) {
            int n_queries = std::get<0>(device_partitions[device_id]).size();
            float ratio = (float) n_queries / (float) n_queries_total;
            int n_threads_device = std::round(ratio * n_threads);

            if (thread_start_idx[device_id].size() < 1) {
                thread_start_idx[device_id].push_back(-1);
                available_threads--;
            }
            else if (thread_start_idx[device_id].size() < n_threads_device) {
                thread_start_idx[device_id].push_back(-1);
                available_threads--;
            }

            device_id = (device_id + 1) % processing_units;
        }

        // Determine the number of queries per thread for each device.
        for (int did = 0; did < processing_units; did++) {
            int n_threads_device = thread_start_idx[did].size();
            int n_queries = std::get<0>(device_partitions[did]).size();
            int stride = (float) n_queries / (float) n_threads_device;

            // Use only a single thread if there are not enough queries to
            // distribute over all threads.
            if (n_queries < n_threads_device) {
                thread_start_idx[did].resize(1);
                thread_start_idx[did][0] = 0;
            }
            else {
                int n_unused_threads = 0;
                // Determine the starting query index for each thread.
                for (int tid = 0; tid < n_threads_device; tid++) {
                    int curr_query = std::get<0>(device_partitions[did])[tid * stride].get_query();
                    int start_index = tid * stride;

                    // Check if the current query has been assigned to a previous
                    // thread. If so, assign the next query to the current thread.
                    bool duplicate = false;
                    if (tid - 1 > 0) {
                        int prev_query = std::get<0>(device_partitions[did])[thread_start_idx[did][tid - 1]].get_query();
                        if (curr_query == prev_query) {
                            duplicate = true;
                            for (int i = start_index; i < n_queries; i++) {
                                // If a query is found that is not assigned to a
                                // previous thread, assign it to the current thread.
                                if (std::get<0>(device_partitions[did])[i].get_query() != curr_query) {
                                    // Reassign the current query and starting index.
                                    duplicate = false;
                                    start_index = i;
                                    curr_query = std::get<0>(device_partitions[did])[i].get_query();
                                    break;
                                }
                            }
                            if (duplicate) {
                                // If no query is found that is not assigned to a
                                // previous thread, increase the number of unused
                                // threads. These will be removed later.
                                n_unused_threads++;
                            }
                            break;
                        }
                    }

                    // Look behind in the partition to find the start of the current
                    // set of similar queries.
                    for (int i = start_index; i >= 0; i--) {
                        if (std::get<0>(device_partitions[did])[i].get_query() != curr_query) {
                            thread_start_idx[did][tid] = i + 1;
                            break;
                        }
                        else if (i == 0) {
                            thread_start_idx[did][tid] = 0;
                        }
                    }
                }

                // Remove unused threads from the current partition and assign them
                // to the next partition, if possible.
                if (n_unused_threads > 0) {
                    thread_start_idx[did].resize(n_threads_device - n_unused_threads);
                    // Add the removed threads to the next partition.
                    if (did < processing_units - 1) {
                        thread_start_idx[did + 1].resize(thread_start_idx[did + 1].size() + n_unused_threads, -1);
                    }
                }
            }
        }
    }


    //-----------------------------------------------------------------------//
    // Allocate memory.                                                      //
    //-----------------------------------------------------------------------//

    auto h2d_init_start_time = std::chrono::high_resolution_clock::now();

    // Allocate memory on the device.
    SearchResult_Dev* dataset_dev[n_devices];
    size_t fmem_dev[processing_units * 2];
    for (int device_id = 0; device_id < processing_units; device_id++) {
        size_t fmem, tmem, fmem_new, tmem_new;

        if (exec_mode == 0 || exec_mode == 2) {
            CUDA_CHECK(cudaSetDevice(device_id));

            // Retrieve avaliable memory in bytes.
            get_device_memory(device_id, fmem, tmem, 1);
            fmem_dev[device_id * 2] = 0; // Memory in use.
            fmem_dev[device_id * 2 + 1] = fmem; // Total available memory.

            // Convert the host-side dataset to a smaller device-side dataset.
            std::vector<SearchResult_Dev> dataset_dev_tmp;
            convert_to_device(std::get<0>(device_partitions[device_id]), dataset_dev_tmp);

            // Check whether the current device has enough free memory available.
            double dataset_size = dataset_dev_tmp.size() * sizeof(SearchResult_Dev);
            if (dataset_size * 1.001 > fmem) {
                Communicate::error_check("[" + std::to_string(node_id) + "] Error: Insufficient GPU memory!\n\tAllocating dataset requires an additional " + std::to_string((dataset_size - fmem_dev[device_id * 2 + 1]) / 1e6) + " MB of GPU memory.");
            }

            // Allocate memory for the dataset on the current device.
            CUDA_CHECK(cudaMalloc(&dataset_dev[device_id], dataset_size));
            CUDA_CHECK(cudaMemcpy(dataset_dev[device_id], dataset_dev_tmp.data(),
                                dataset_size, cudaMemcpyHostToDevice));
            dataset_dev_tmp.clear();

            fmem_dev[device_id * 2] += dataset_size;

            // Allocate memory for the query dependent parameters on both the current device and host.
            cm_hosts[device_id]->init_parameters(device_partitions[device_id], fmem_dev[device_id * 2 + 1] - fmem_dev[device_id * 2], true);
            Communicate::error_check();
            fmem_dev[device_id * 2] += cm_hosts[device_id]->get_memory_usage();

            // Show memory usage.
            get_device_memory(device_id, fmem_new, tmem_new, 1);
        }
        else {
            // Retrieve avaliable memory in bytes.
            get_host_memory(fmem, tmem, 1);

            fmem_dev[device_id * 2] = 0; // Memory in use.
            fmem_dev[device_id * 2 + 1] = fmem; // Total available memory.

            // Check whether the current device has enough free memory available.
            double dataset_size = std::get<0>(device_partitions[device_id]).size() * sizeof(SERP_Hst);

            fmem_dev[device_id * 2] += dataset_size;
            if (dataset_size * 1.001 > fmem) {
                Communicate::error_check("[" + std::to_string(node_id) + "] Error: Insufficient system memory!\n\tAllocating dataset requires an additional " + std::to_string((dataset_size - fmem_dev[device_id * 2 + 1]) / 1e6) + " MB of system memory.");
            }

            // Allocate memory for the query dependent parameters on both the current device and host.
            cm_hosts[device_id]->init_parameters(device_partitions[device_id], fmem_dev[device_id * 2 + 1] - fmem_dev[device_id * 2], false);
            Communicate::error_check();
            fmem_dev[device_id * 2] += cm_hosts[device_id]->get_memory_usage();

            // Show memory usage.
            get_host_memory(fmem_new, tmem_new, 1);
        }

        std::cout << "[" << node_id << ", " << ((exec_mode == 0 || exec_mode == 2) ? device_id : -1) << "], expected memory usage = " <<
        fmem_dev[device_id * 2] / 1e6 << "/" << fmem_dev[device_id * 2 + 1] / 1e6 << " MB (" <<
        (int) ((float) fmem_dev[device_id * 2] / (float) fmem_dev[device_id * 2 + 1] * 100) << "%)\n" <<
        "\tmeasured memory usage = " << (fmem - fmem_new) / 1e6 << "/" << fmem_dev[device_id * 2 + 1] / 1e6 << " MB (" <<
        (int) ((float) (fmem - fmem_new) / (float) fmem_dev[device_id * 2 + 1] * 100) << "%)" << std::endl;
    }

    // Wait for all nodes to finish allocating memory and check if any node
    // raised an error while doing so.
    Communicate::error_check();
    Communicate::barrier();

    auto h2d_init_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> h2d_init_elapsed_time = h2d_init_stop_time - h2d_init_start_time;


    //-----------------------------------------------------------------------//
    // Initate device-side click model.                                      //
    //-----------------------------------------------------------------------//

    // Initialize the device-side click model.
    for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));

        // Retrieve the device-side click model parameter arrays and their sizes.
        Param** parameter_references;
        int* parameter_sizes;
        cm_hosts[device_id]->get_device_references(parameter_references, parameter_sizes);

        // Launch the click model initialization kernel.
        Kernel::initialize<<<1, 1>>>(model_type, node_id, device_id, parameter_references, parameter_sizes);

        CUDA_CHECK(cudaDeviceSynchronize());
    }


    //-----------------------------------------------------------------------//
    // Estimate CM parameters n_itr times.                                   //
    //-----------------------------------------------------------------------//

    // Initiate CUDA event timers.
    cudaEvent_t start_events[n_devices], end_events[n_devices];
    double avg_time_comp{0}, avg_time_update{0};
    std::chrono::duration<double> tot_time_em{0}, avg_time_itr{0}, avg_time_sync{0}, avg_time_h2d{0}, avg_time_d2h{0};

    for (int dev = 0; dev < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); dev++) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaEventCreate(&start_events[dev]));
        CUDA_CHECK(cudaEventCreate(&end_events[dev]));
    }

    // Get kernel dimensions.
    int kernel_dims[n_devices * 2];
    for (int did = 0; did < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); did++) {
        int n_queries = std::get<0>(device_partitions[did]).size(); // Number of non-unique queries in the dataset.
        // Number of threads per block.
        int block_size = BLOCK_SIZE;
        // Calculate the number of blocks in which the array size can be split
        // up. (block_size - 1) is used to ensure that there won't be an
        // insufficient amount of blocks.
        kernel_dims[did * 2] = (n_queries + (block_size - 1)) / block_size;
        kernel_dims[did * 2 + 1] = block_size;

        std::cout << "[" << node_id << ", " << did << "], kernel dimensions = <<<" << kernel_dims[did * 2] << ", " << kernel_dims[did * 2 + 1] << ">>>" << std::endl;
    }

    if (node_id == ROOT) {
        std::cout << "\nStarting " << n_itr << " EM parameter estimation iterations..." << std::endl;
    }

    // Perform n_itr Expectation-Maximization iterations.
    for (int itr = 0; itr < n_itr; itr++) {

        //-------------------------------------------------------------------//
        // Launch parameter estimation kernel.                               //
        //-------------------------------------------------------------------//

        auto em_itr_start_time = std::chrono::high_resolution_clock::now();
        double em_comp_elapsed_time{0};

         // GPU-only.
        for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));

            int grid_size = kernel_dims[device_id * 2];
            int block_size = kernel_dims[device_id * 2 + 1];
            int dataset_size = std::get<0>(device_partitions[device_id]).size();

            CUDA_CHECK(cudaEventRecord(start_events[device_id], 0));
            Kernel::em_training<<<grid_size, block_size>>>(dataset_dev[device_id], dataset_size);
            CUDA_CHECK(cudaEventRecord(end_events[device_id], 0));
        }

        for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_events[device_id], end_events[device_id]));
            em_comp_elapsed_time += (double) ((time_ms / 1000.f) / ((double) n_devices));
        }

        if (exec_mode == 1) { // CPU-only.
            auto em_comp_start_time = std::chrono::high_resolution_clock::now();
            for (int unit = 0; unit < processing_units; unit++) {
                cm_hosts[unit]->process_session(std::get<0>(device_partitions[unit]), thread_start_idx[unit]);
            }
            em_comp_elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - em_comp_start_time).count();
        }

        avg_time_comp += em_comp_elapsed_time / ((double) n_itr);


        //-------------------------------------------------------------------//
        // Wipe previous parameter results.                                  //
        //-------------------------------------------------------------------//

        for (int device_id = 0; device_id < processing_units; device_id++) {
            if (exec_mode == 0 || exec_mode == 2) { CUDA_CHECK(cudaSetDevice(device_id)); }

            auto h2d_start_time = std::chrono::high_resolution_clock::now();

            cm_hosts[device_id]->reset_parameters((exec_mode == 0 || exec_mode == 2) ? true : false);

            auto h2d_stop_time = std::chrono::high_resolution_clock::now();
            avg_time_h2d += (h2d_stop_time - h2d_start_time) / n_itr;
        }

        for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) { // GPU-only.
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaDeviceSynchronize());
        }


        //-------------------------------------------------------------------//
        // Launch parameter update kernel.                                   //
        //-------------------------------------------------------------------//

        double em_update_elapsed_time{0};
        // GPU-only.
        for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));

            int grid_size = kernel_dims[device_id * 2];
            int block_size = kernel_dims[device_id * 2 + 1];
            int dataset_size = std::get<0>(device_partitions[device_id]).size();

            CUDA_CHECK(cudaEventRecord(start_events[device_id], 0));
            Kernel::update<<<grid_size, block_size>>>(dataset_dev[device_id], dataset_size);

            CUDA_CHECK(cudaEventRecord(end_events[device_id], 0));
        }

        for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_events[device_id], end_events[device_id]));
            em_update_elapsed_time += (double) ((time_ms / 1000.f) / ((double) n_devices));
        }

        if (exec_mode == 1) { // CPU-only.
            auto em_update_start_time = std::chrono::high_resolution_clock::now();
            for (int unit = 0; unit < processing_units; unit++) {
                cm_hosts[unit]->update_parameters(std::get<0>(device_partitions[unit]), thread_start_idx[unit]);
            }
            em_update_elapsed_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - em_update_start_time).count();
        }

        avg_time_update += em_update_elapsed_time / n_itr;


        //-------------------------------------------------------------------//
        // Synchronize parameters across the nodes and devices.              //
        //-------------------------------------------------------------------//

        auto em_sync_itr_start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<std::vector<Param>>> public_parameters(processing_units); // Device ID -> Parameter type -> Parameters.

        // Retrieve all types of public parameters from each device.
        for (int device_id = 0; device_id < processing_units; device_id++) {
            auto d2h_start_time = std::chrono::high_resolution_clock::now();

            if (exec_mode == 0 || exec_mode == 2) {
                CUDA_CHECK(cudaSetDevice(device_id));
                cm_hosts[device_id]->transfer_parameters(PUBLIC, D2H, false);
            }

            auto d2h_stop_time = std::chrono::high_resolution_clock::now();
            avg_time_d2h += (d2h_stop_time - d2h_start_time) / n_itr;

            cm_hosts[device_id]->get_parameters(public_parameters[device_id], PUBLIC);
        }

        // Synchronize the parameters local to this device before synchronizing
        // with the parameters from other nodes.
        Communicate::sync_parameters(public_parameters);

        // Send this node's synchronized public device parameters to all other
        // nodes in the network.
        std::vector<std::vector<std::vector<Param>>> network_parameters(n_nodes,
            std::vector<std::vector<Param>>(public_parameters.size(),
            std::vector<Param>(public_parameters[0][0].size()))); // Node ID -> Parameter type -> Parameters.
        Communicate::exchange_parameters(network_parameters, public_parameters[0], n_nodes, node_id);

        // Sychronize the public parameters received from other nodes.
        Communicate::sync_parameters(network_parameters);

        // Move all types of synchronized public parameters back to each device.
        for (int device_id = 0; device_id < processing_units; device_id++) {
            cm_hosts[device_id]->set_parameters(network_parameters[0], PUBLIC);

            auto h2d_start_time = std::chrono::high_resolution_clock::now();

            if (exec_mode == 0 || exec_mode == 2) {
                CUDA_CHECK(cudaSetDevice(device_id));
                cm_hosts[device_id]->transfer_parameters(PUBLIC, H2D, false);
            }

            auto h2d_stop_time = std::chrono::high_resolution_clock::now();
            avg_time_h2d += (h2d_stop_time - h2d_start_time) / n_itr;
        }

        auto em_sync_itr_stop_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> em_sync_elapsed_time = em_sync_itr_stop_time - em_sync_itr_start_time;
        avg_time_sync += em_sync_elapsed_time / n_itr;

        auto em_itr_stop_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> em_itr_elapsed_time = em_itr_stop_time - em_itr_start_time;
        avg_time_itr += em_itr_elapsed_time / n_itr;
        tot_time_em += em_itr_elapsed_time;

        // Show metrics on the root node.
        if (node_id == ROOT) {
            int itr_len = std::to_string(n_itr).length();
            std::cout << "Itr: " << std::left << std::setw(itr_len) << itr <<
            " Itr-time: " << std::left << std::setw(10) << em_itr_elapsed_time.count() <<
            " Itr-EM_COMP: " << std::left << std::setw(11) << em_comp_elapsed_time <<
            " Itr-EM_UPDATE: " << std::left << std::setw(10) << em_update_elapsed_time <<
            " Itr-Sync: " << std::left << std::setw(12) << em_sync_elapsed_time.count() << std::endl;
        }
    }

    // Destroy CUDA timer events.
    for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
        CUDA_CHECK(cudaEventDestroy(start_events[device_id]));
        CUDA_CHECK(cudaEventDestroy(end_events[device_id]));
    }


    //-----------------------------------------------------------------------//
    // Copy trained (partial) click model from the device to the host.       //
    //-----------------------------------------------------------------------//

    for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));

        // Ensure that all kernels have finished their execution.
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy all device-side parameters to host.
        cm_hosts[device_id]->transfer_parameters(ALL, D2H, false);
    }


    //-----------------------------------------------------------------------//
    // Evaluate CM using log-likelihood and perplexity on each node.         //
    //-----------------------------------------------------------------------//

    auto em_eval_start_time = std::chrono::high_resolution_clock::now();

    // Calculate node-local log-likelihood and perplexity.
    std::map<int, std::array<float, 2>> llh_device;
    std::map<int, Perplexity> ppl_device;
    for (int device_id = 0; device_id < processing_units; device_id++) {
        // Compute the log-likelihood.
        LogLikelihood llh(cm_hosts[device_id]);
        float llh_val = llh.evaluate(std::get<1>(device_partitions[device_id]));
        int task_size = std::get<1>(device_partitions[device_id]).size();
        std::array<float, 2> temp_arr{llh_val, static_cast<float>(task_size)};
        llh_device[device_id] = temp_arr;

        // Compute the perplexity.
        Perplexity ppl;
        ppl.evaluate(cm_hosts[device_id], std::get<1>(device_partitions[device_id]));
        ppl_device[device_id] = ppl;
    }

    // Gather the log-likelihood and perplexity of all nodes on the root node.
    Communicate::gather_evaluations(llh_device, ppl_device, n_nodes, node_id, n_devices_network);

    auto em_eval_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> em_eval_elapsed_time = em_eval_stop_time - em_eval_start_time;


    //-----------------------------------------------------------------------//
    // Free allocated device-side memory.                                    //
    //-----------------------------------------------------------------------//

    for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dataset_dev[device_id]));
        cm_hosts[device_id]->destroy_parameters();
    }


    //-----------------------------------------------------------------------//
    // Output trained click model.                                           //
    //-----------------------------------------------------------------------//

    if (!output_path.empty()) {
        std::vector<QDP> qdp_output;
        std::ofstream output_file;

        // Create a new text file for the output on the root node.
        if (node_id == ROOT) {
            output_file.open(output_path);
            output_file << "query,document,probability" << std::endl;
        }

        // Fill the qdp_output vector with the query-document pairs.
        auto get_qdp_output = [](std::vector<SERP_Hst>& dataset, std::vector<QDP>& qdp_output, ClickModel_Hst& cm_host, int start_idx, int stop_idx) {
            qdp_output.resize((stop_idx - start_idx) * MAX_SERP);
            float probabilities[MAX_SERP];
            for (int i = start_idx, j = 0; i < stop_idx; i++) {
                SERP_Hst query_session = dataset[i];
                int query = query_session.get_query();
                cm_host.get_serp_probability(query_session, probabilities);
                for (int rank = 0; rank < MAX_SERP; rank++) {
                    int document = query_session[rank].get_doc_id();
                    qdp_output[j].query = query;
                    qdp_output[j].document = document;
                    qdp_output[j].probability = probabilities[rank];
                    j++;
                }
            }
        };

        // Send the query-document pairs and their probability to the root node
        // and write it there to the output file.
        if (node_id == ROOT) {
            std::cout << "\nWriting output to file..." << std::endl;

            // Iterate over all nodes' results and write them to a file.
            for (int nid = 1; nid < n_nodes; nid++) {
                for (int did = 0; did < n_devices_network[nid]; did++) {
                    // Receive the queries, documents, and probabilities from
                    // the node.
                    Communicate::gather_results(node_id, nid, qdp_output);

                    // Write the received results to a file.
                    for (auto QDP : qdp_output) {
                        output_file << QDP.query << "," << QDP.document << "," << QDP.probability << std::endl;
                    }
                }
            }
        }

        // Iterate over the datasets assigned to each of this node's devices.
        for (int unit = 0; unit < processing_units; unit++) {
            // Fill the qdp_output vector with the probability of each
            // query-document pair in a part of the dataset being clicked.
            get_qdp_output(std::get<0>(device_partitions[unit]), qdp_output,
                            *cm_hosts[unit], 0, std::get<0>(device_partitions[unit]).size());

            if (node_id != ROOT) { // Have the root node write the results to a file.
                // Send the qdp_output vector to the root node.
                Communicate::gather_results(node_id, ROOT, qdp_output);
            }
            else { // Write the resuts to a file on the root node.
                for (auto QDP : qdp_output) {
                    output_file << QDP.query << "," << QDP.document << "," << QDP.probability << std::endl;
                }
            }
        }

        if (node_id == ROOT) {
            // Close the file.
            output_file.close();
        }
    }


    //-----------------------------------------------------------------------//
    // Show metrics.                                                         //
    //-----------------------------------------------------------------------//

    // Show metrics on the root node.
    if (node_id == ROOT) {
        // Compute the total log-likelihood.
        float total_llh_sum = 0.0;
        float total_llh_sessions = 0.0;
        std::for_each(llh_device.begin(), llh_device.end(), [&] (std::pair<const int, std::array<float, 2>>& llh_task) {
            total_llh_sum += llh_task.second[0];
            total_llh_sessions += llh_task.second[1];
        });

        std::cout << "\nTotal Log likelihood is: " << total_llh_sum / total_llh_sessions << std::endl;

        float total_task_size{0.0};
        std::array<float, 10> temp_task_rank_perplexities{0.0};

        // Compute perplexity at every rank.
        for (auto const& itr: ppl_device){
            total_task_size += itr.second.task_size;
            for (int j{0}; j < 10; j++){
                temp_task_rank_perplexities[j] += itr.second.task_rank_perplexities[j];
            }
        }

        std::array<float, 10> ppl_per_rank{};
        float ppl_value;
        for (int i{0}; i < 10; i++){
            ppl_per_rank[i] = std::pow(2, (-1 * temp_task_rank_perplexities[i])/total_task_size);
            std::cout << "Perplexity at rank " << i << " is: " << ppl_per_rank[i] << std::endl;
        }
        ppl_value = std::accumulate(ppl_per_rank.begin(), ppl_per_rank.end(), 0.0) / 10.0;
        std::cout << "Perplexity is: " << ppl_value << std::endl;

        // Show the timing measurements of the EM algorithm.
        if (exec_mode == 0 || exec_mode == 2) {
            std::cout << "\nHost to Device dataset transfer time: " << h2d_init_elapsed_time.count() <<
            "\nAverage Host to Device parameter transfer time: " << avg_time_h2d.count() <<
            "\nAverage Device to Host parameter transfer time: " << avg_time_d2h.count() << std::endl;
        }

        std::cout << "\nAverage time per iteration: " << avg_time_itr.count() <<
        "\nAverage time per computation in each iteration: " << avg_time_comp <<
        "\nAverage time per update in each iteration: " << avg_time_update <<
        "\nAverage time per synchronization in each iteration: " << avg_time_sync.count() <<
        "\nTotal time of training: " << tot_time_em.count() <<
        "\nEvaluation time: " << em_eval_elapsed_time.count() << std::endl;
    }

    // Destroy all allocations on all available devices as part of the shutdown
    // procedure.
    for (int device_id = 0; device_id < ((exec_mode == 0 || exec_mode == 2) ? n_devices : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaDeviceReset());
    }
}