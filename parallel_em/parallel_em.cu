/** Parallelizing EM on GPU(s).
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * parallel_em.cu:
 *  - Defines the functions used for initiating the EM process on the GPU.
 */

// User include.
#include "parallel_em.cuh"

#define GPU_EXECUTION(exec_mode) (exec_mode == 0 || exec_mode == 2)

//---------------------------------------------------------------------------//
// Host-side computation.                                                    //
//---------------------------------------------------------------------------//

/**
 * @brief Execute EM algorithm in parallel.
 *
 * @param config: Processing configuration.
 * @param device_partitions: The dataset assigned to each of the node's devices.
 * @param output_path: The path where to store the output.
 * @param hostname: The hostname of the device.
 */
 void em_parallel(
    const ProcessingConfig& config,
    LocalPartitions& device_partitions,
    const std::string& output_path,
    const char* hostname
) {
    Timer timer;

    if (config.node_id == ROOT) {
        std::cout << "\nExpectation Maximization (EM) in parallel ..." << std::endl;
    }

    //-----------------------------------------------------------------------//
    // Initate host-side click model.                                        //
    //-----------------------------------------------------------------------//

    // Initiate a host-side click model for each device or for the host.
    ClickModel_Hst* cm_hosts[config.unit_count];
    for (int unit = 0; unit < config.unit_count; unit++) {
        // Initialize the click model.
        cm_hosts[unit] = create_cm_host(config.model_type);
        // Print a confirmation message on the first device of the root node.
        if (config.node_id == ROOT && unit == 0) {
            cm_hosts[unit]->say_hello();
        }
    }

    //-----------------------------------------------------------------------//
    // Assign queries to CPU threads.                                        //
    //-----------------------------------------------------------------------//

    std::vector<std::vector<int>> thread_start_idx(config.unit_count);
    if (config.exec_mode != 0) {
        int n_queries_total = std::accumulate(device_partitions.begin(), device_partitions.end(), 0,
            [](int sum, const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition) {
            return sum + std::get<0>(partition).size();});
        int available_threads = config.thread_count;

        // Allocate CPU threads to each device based on the number of assigned
        // queries to the device.
        int device_id = 0; // Start thread assignment with device 0.
        while (available_threads > 0) {
            int n_queries = std::get<0>(device_partitions[device_id]).size();
            float ratio = (float) n_queries / (float) n_queries_total;
            int n_threads_per_device = std::round(ratio * config.thread_count);

            // Ensure that each device always gets at least one thread and that
            // the assigned threads do not exceed the available threads.
            if (thread_start_idx[device_id].size() < 1 || thread_start_idx[device_id].size() < n_threads_per_device) {
                thread_start_idx[device_id].push_back(-1);
                available_threads--;
            }

            // Select next device.
            device_id = (device_id + 1) % config.unit_count;
        }

        // Determine the number of queries per thread for each device.
        for (int did = 0; did < config.unit_count; did++) {
            int n_threads_per_device = thread_start_idx[did].size();
            int n_queries = std::get<0>(device_partitions[did]).size();
            int stride = (float) n_queries / (float) n_threads_per_device; // Approximate number of queries per thread.

            // Use only a single thread if there are not enough queries to
            // distribute over all threads.
            if (n_queries < n_threads_per_device) {
                thread_start_idx[did].resize(1);
                thread_start_idx[did][0] = 0;
            }
            else {
                int n_unused_threads = 0;
                // Determine the starting query index for each thread.
                for (int tid = 0; tid < n_threads_per_device; tid++) {
                    int curr_query = std::get<0>(device_partitions[did])[tid * stride].get_query();
                    int start_index = tid * stride;

                    // Check if the current query has been assigned to a previous
                    // thread. If so, assign the next query to the current thread.
                    bool duplicate = false;
                    if (tid - 1 > 0) { // Check if there are any threads before the current thread.
                        int prev_query = std::get<0>(device_partitions[did])[thread_start_idx[did][tid - 1]].get_query();

                        // If the current query has already been assigned, then
                        // find the next available query in the dataset.
                        if (curr_query == prev_query) {
                            duplicate = true;
                            // Search through the remaining queries.
                            for (int i = start_index; i < n_queries; i++) {
                                // If a query is found that is not assigned to a
                                // previous thread, assign it to the current thread.
                                if (std::get<0>(device_partitions[did])[i].get_query() != curr_query) {
                                    // Reassign this thread's query and
                                    // starting index.
                                    duplicate = false;
                                    start_index = i;
                                    curr_query = std::get<0>(device_partitions[did])[i].get_query();
                                    break;
                                }
                            }

                            // If no query is found that is not assigned to a
                            // previous thread, increase the number of unused
                            // threads. These will be reassigned later.
                            if (duplicate) {
                                n_unused_threads++;
                            }
                        }
                    }

                    // Find the start of the current set of similar queries in
                    // this device's dataset.
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

                // Remove unused threads from the current partition and assign
                // them to the next partition, if possible.
                if (n_unused_threads > 0) {
                    thread_start_idx[did].resize(n_threads_per_device - n_unused_threads);
                    // Add the removed threads to the next partition.
                    if (did < config.unit_count - 1) {
                        thread_start_idx[did + 1].resize(thread_start_idx[did + 1].size() + n_unused_threads, -1);
                    }
                }
            }
        }
    }


    //-----------------------------------------------------------------------//
    // Allocate memory.                                                      //
    //-----------------------------------------------------------------------//

    timer.start("h2d_init");

    // Allocate memory on the device.
    SearchResult_Dev* dataset_dev[config.device_count];
    size_t fmem_dev[config.unit_count * 2];
    for (int device_id = 0; device_id < config.unit_count; device_id++) {
        size_t fmem, tmem, fmem_new, tmem_new;

        // Allocate memory on either the device or the host.
        if (GPU_EXECUTION(config.exec_mode)) {
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
                Communicate::error_check("[" + std::string(hostname) + "] \033[12;31mError\033[0m: Insufficient GPU memory!\n\tAllocating dataset requires an additional " + std::to_string((dataset_size - fmem_dev[device_id * 2 + 1]) / 1e6) + " MB of GPU memory.");
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

            // Dataset size does not need to be checked, since the dataset has
            // already been allocated on the host.

            // Allocate memory for the query dependent parameters on both the current device and host.
            cm_hosts[device_id]->init_parameters(device_partitions[device_id], fmem_dev[device_id * 2 + 1] - fmem_dev[device_id * 2], false);
            Communicate::error_check();
            fmem_dev[device_id * 2] += cm_hosts[device_id]->get_memory_usage();

            // Show memory usage.
            get_host_memory(fmem_new, tmem_new, 1);
        }

        // Show memory usage.
        int device = (GPU_EXECUTION(config.exec_mode)) ? device_id : -1;
        float expected_mem_usage = fmem_dev[device_id * 2] / 1e6;
        float total_mem_capacity = fmem_dev[device_id * 2 + 1] / 1e6;
        int expected_mem_percent = static_cast<int>(expected_mem_usage / total_mem_capacity * 100);

        float measured_mem_usage = (fmem - fmem_new) / 1e6;
        int measured_mem_percent = static_cast<int>(measured_mem_usage / total_mem_capacity * 100);

        std::ostringstream output;
        output << "[" << hostname << ", " << device << "], memory expected = " << expected_mem_usage << "/" << total_mem_capacity << " MB (" << expected_mem_percent << "%), measured = " << measured_mem_usage << "/" << total_mem_capacity << " MB (" << measured_mem_percent << "%)";
        std::cout << output.str() << std::endl;
    }

    // Wait for all nodes to finish allocating memory and check if any node
    // raised an error while doing so.
    Communicate::error_check();
    Communicate::barrier();

    timer.stop("h2d_init");


    //-----------------------------------------------------------------------//
    // Initate device-side click model.                                      //
    //-----------------------------------------------------------------------//

    // Initialize the device-side click model.
    for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));

        // Retrieve the device-side click model parameter arrays and their sizes.
        Param** parameter_references;
        int* parameter_sizes;
        cm_hosts[device_id]->get_device_references(parameter_references, parameter_sizes);

        // Launch the click model initialization kernel.
        Kernel::initialize<<<1, 1>>>(config.model_type, config.node_id, device_id, parameter_references, parameter_sizes);

        CUDA_CHECK(cudaDeviceSynchronize());
    }


    //-----------------------------------------------------------------------//
    // Estimate CM parameters n_itr times.                                   //
    //-----------------------------------------------------------------------//

    // Initiate CUDA event timers.
    cudaEvent_t start_events[config.device_count], end_events[config.device_count];

    for (int dev = 0; dev < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); dev++) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaEventCreate(&start_events[dev]));
        CUDA_CHECK(cudaEventCreate(&end_events[dev]));
    }

    // Get kernel dimensions.
    int kernel_dims[config.device_count * 2];
    for (int did = 0; did < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); did++) {
        int n_queries = std::get<0>(device_partitions[did]).size(); // Number of non-unique queries in the dataset.
        // Number of threads per block.
        int block_size = BLOCK_SIZE;
        // Calculate the number of blocks in which the array size can be split
        // up. (block_size - 1) is used to ensure that there won't be an
        // insufficient amount of blocks.
        kernel_dims[did * 2] = (n_queries + (block_size - 1)) / block_size;
        kernel_dims[did * 2 + 1] = block_size;

        std::cout << "[" << hostname << ", " << did << "] kernel dimensions = <<<" << kernel_dims[did * 2] << ", " << kernel_dims[did * 2 + 1] << ">>>" << std::endl;
    }

    if (config.node_id == ROOT) {
        std::cout << "\nStarting " << config.iterations << " EM parameter estimation iterations..." << std::endl;
    }

    // Perform n_itr Expectation-Maximization iterations.
    timer.start("EM iteration");
    for (int itr = 0; itr < config.iterations; itr++) {

        //-------------------------------------------------------------------//
        // Launch parameter estimation kernel.                               //
        //-------------------------------------------------------------------//

        timer.start("EM computation");

         // GPU-only.
        for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));

            int grid_size = kernel_dims[device_id * 2];
            int block_size = kernel_dims[device_id * 2 + 1];
            int dataset_size = std::get<0>(device_partitions[device_id]).size();

            CUDA_CHECK(cudaEventRecord(start_events[device_id], 0));
            Kernel::em_training<<<grid_size, block_size>>>(dataset_dev[device_id], dataset_size);
            CUDA_CHECK(cudaEventRecord(end_events[device_id], 0));
        }

        for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaDeviceSynchronize());
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_events[device_id], end_events[device_id]));
        }

        if (config.exec_mode == 1) { // CPU-only.
            for (int unit = 0; unit < config.unit_count; unit++) {
                cm_hosts[unit]->process_session(std::get<0>(device_partitions[unit]), thread_start_idx[unit]);
            }
        }

        timer.lap("EM computation", false);


        //-------------------------------------------------------------------//
        // Wipe previous parameter results.                                  //
        //-------------------------------------------------------------------//

        for (int device_id = 0; device_id < config.unit_count; device_id++) {
            if (GPU_EXECUTION(config.exec_mode)) { CUDA_CHECK(cudaSetDevice(device_id)); }
            timer.start("h2d");
            cm_hosts[device_id]->reset_parameters((GPU_EXECUTION(config.exec_mode)) ? true : false);
            timer.lap("h2d", false);
        }

        for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) { // GPU-only.
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaDeviceSynchronize());
        }


        //-------------------------------------------------------------------//
        // Launch parameter update kernel.                                   //
        //-------------------------------------------------------------------//

        timer.start("EM update");

        // GPU-only.
        for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));

            int grid_size = kernel_dims[device_id * 2];
            int block_size = kernel_dims[device_id * 2 + 1];
            int dataset_size = std::get<0>(device_partitions[device_id]).size();
            size_t shr_mem_size = BLOCK_SIZE * MAX_SERP * sizeof(int);

            CUDA_CHECK(cudaEventRecord(start_events[device_id], 0));
            Kernel::update<<<grid_size, block_size, shr_mem_size>>>(dataset_dev[device_id], dataset_size);
            CUDA_CHECK(cudaEventRecord(end_events[device_id], 0));
        }

        for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
            CUDA_CHECK(cudaSetDevice(device_id));
            CUDA_CHECK(cudaDeviceSynchronize());
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_events[device_id], end_events[device_id]));
        }

        if (config.exec_mode == 1) { // CPU-only.
            for (int unit = 0; unit < config.unit_count; unit++) {
                cm_hosts[unit]->update_parameters(std::get<0>(device_partitions[unit]), thread_start_idx[unit]);
            }
        }

        timer.lap("EM update", false);


        //-------------------------------------------------------------------//
        // Synchronize parameters across the nodes and devices.              //
        //-------------------------------------------------------------------//

        timer.start("EM synchronization");
        timer.start("d2h");

        std::vector<std::vector<std::vector<Param>>> public_parameters(config.unit_count); // Device ID -> Parameter type -> Parameters.

        // Retrieve all types of public parameters from each device.
        for (int device_id = 0; device_id < config.unit_count; device_id++) {
            if (GPU_EXECUTION(config.exec_mode)) {
                CUDA_CHECK(cudaSetDevice(device_id));
                cm_hosts[device_id]->transfer_parameters(PUBLIC, D2H, false);
            }

            cm_hosts[device_id]->get_parameters(public_parameters[device_id], PUBLIC);
        }
        timer.lap("d2h", false);

        // Synchronize the parameters local to this device before synchronizing
        // with the parameters from other nodes.
        Communicate::sync_parameters(public_parameters);

        // Send this node's synchronized public device parameters to all other
        // nodes in the network.
        std::vector<std::vector<std::vector<Param>>> network_parameters(config.total_nodes,
            std::vector<std::vector<Param>>(public_parameters.size(),
            std::vector<Param>(public_parameters[0][0].size()))); // Node ID -> Parameter type -> Parameters.
        Communicate::exchange_parameters(network_parameters, public_parameters[0], config.total_nodes, config.node_id);

        // Sychronize the public parameters received from other nodes.
        Communicate::sync_parameters(network_parameters);

        // Move all types of synchronized public parameters back to each device.
        for (int device_id = 0; device_id < config.unit_count; device_id++) {
            cm_hosts[device_id]->set_parameters(network_parameters[0], PUBLIC);

            timer.start("h2d");

            if (GPU_EXECUTION(config.exec_mode)) {
                CUDA_CHECK(cudaSetDevice(device_id));
                cm_hosts[device_id]->transfer_parameters(PUBLIC, H2D, false);
            }

            timer.lap("h2d");
        }

        timer.lap("EM synchronization", false);

        // Show metrics on the root node.
        if (config.node_id == ROOT) {
            int itr_len = std::to_string(config.iterations).length();
            std::cout << "Itr: " << std::left << std::setw(itr_len) << itr <<
            " Itr-time: " << std::left << std::setw(10) << timer.lap("EM iteration") <<
            " Itr-EM_COMP: " << std::left << std::setw(11) << timer.elapsed("EM computation") <<
            " Itr-EM_UPDATE: " << std::left << std::setw(10) << timer.elapsed("EM update") <<
            " Itr-Sync: " << std::left << std::setw(12) << timer.elapsed("EM synchronization") << std::endl;
        }
    }

    // Destroy CUDA timer events.
    for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
        CUDA_CHECK(cudaEventDestroy(start_events[device_id]));
        CUDA_CHECK(cudaEventDestroy(end_events[device_id]));
    }


    //-----------------------------------------------------------------------//
    // Copy trained (partial) click model from the device to the host.       //
    //-----------------------------------------------------------------------//

    for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));

        // Ensure that all kernels have finished their execution.
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy all device-side parameters to host.
        cm_hosts[device_id]->transfer_parameters(ALL, D2H, false);
    }


    //-----------------------------------------------------------------------//
    // Evaluate CM using log-likelihood and perplexity on each node.         //
    //-----------------------------------------------------------------------//

    timer.start("EM evaluation");

    // Calculate node-local log-likelihood and perplexity.
    std::map<int, std::array<float, 2>> llh_device;
    std::map<int, Perplexity> ppl_device;
    for (int device_id = 0; device_id < config.unit_count; device_id++) {
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
    Communicate::gather_evaluations(llh_device, ppl_device, config.total_nodes, config.node_id, config.devices_per_node);

    timer.stop("EM evaluation");


    //-----------------------------------------------------------------------//
    // Free allocated device-side memory.                                    //
    //-----------------------------------------------------------------------//

    for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dataset_dev[device_id]));
        cm_hosts[device_id]->destroy_parameters();
    }


    //-----------------------------------------------------------------------//
    // Output trained click model.                                           //
    //-----------------------------------------------------------------------//

    if (!output_path.empty()) {
        std::cout << "\nWriting output to file..." << std::endl;

        std::pair<std::vector<std::string>, std::vector<std::string>> headers;
        std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> parameters[config.unit_count];
        for (int device_id = 0; device_id < config.unit_count; device_id++) {
            cm_hosts[device_id]->get_parameter_information(headers, parameters[device_id]);
        }
        Communicate::output_parameters(config.node_id, config.unit_count, output_path, device_partitions, headers, parameters);
    }


    //-----------------------------------------------------------------------//
    // Show metrics.                                                         //
    //-----------------------------------------------------------------------//

    // Show metrics on the root node.
    if (config.node_id == ROOT) {
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
        if (GPU_EXECUTION(config.exec_mode)) {
            std::cout << "\nHost to Device dataset transfer time: " << timer.elapsed("h2d_init") <<
            "\nAverage Host to Device parameter transfer time: " << timer.avg("h2d") / 2 <<
            "\nAverage Device to Host parameter transfer time: " << timer.avg("d2h") << std::endl;
        }

        std::cout << "\nAverage time per iteration: " << timer.avg("EM iteration") <<
        "\nAverage time per computation in each iteration: " << timer.avg("EM computation") <<
        "\nAverage time per update in each iteration: " << timer.avg("EM update") <<
        "\nAverage time per synchronization in each iteration: " << timer.avg("EM synchronization") <<
        "\nTotal time of training: " << timer.total("EM iteration") <<
        "\nEvaluation time: " << timer.elapsed("EM evaluation") << std::endl;
    }

    // Destroy all allocations on all available devices as part of the shutdown
    // procedure.
    for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaDeviceReset());
    }
}