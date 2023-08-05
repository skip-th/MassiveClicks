/** Parallelizing EM on GPU(s).
 *
 * parallel_em.cu:
 *  - Defines the functions used for initiating the EM process on the GPU.
 */

// User include.
#include "parallel_em.cuh"

#define GPU_EXECUTION(exec_mode) (exec_mode == 0 || exec_mode == 2)

//---------------------------------------------------------------------------//
// Assign queries to CPU threads.                                            //
//---------------------------------------------------------------------------//

/**
 * @brief Find the next non-duplicate query.
 *
 * @param device_id The ID of the device.
 * @param curr_query The current query.
 * @param start_index The search start index.
 * @param device_partitions The dataset assigned to each of the node's devices.
 * @param thread_start_idx The start index of the thread.
 * @param thread_id The ID of the thread.
 * @return int 1 if the query is a duplicate, 0 otherwise.
 */
int find_next_non_duplicate_query(int device_id, int curr_query, int& start_index,
                                  const LocalPartitions& device_partitions,
                                  std::vector<std::vector<int>>& thread_start_idx, int thread_id) {
    bool duplicate = 1;
    int partition_query_count = std::get<0>(device_partitions[device_id]).size();

    // Search through the remaining queries.
    for (int i = start_index; i < partition_query_count; i++) {
        // If a query is found that is not assigned to a previous thread,
        // assign it to the current thread.
        if (std::get<0>(device_partitions[device_id])[i].get_query() != curr_query) {
            // Reassign this thread's query and starting index.
            duplicate = 0;
            start_index = i;
            curr_query = std::get<0>(device_partitions[device_id])[i].get_query();
            break;
        }
    }
    return duplicate;
}

/**
 * @brief Find the start of the current set of similar queries in this device's
 * dataset.
 *
 * @param device_id The ID of the device.
 * @param curr_query The current query.
 * @param start_index The start index of the thread.
 * @param thread_start_idx The start index of each thread.
 * @param thread_id The ID of the thread.
 * @param device_partitions The dataset assigned to each of the node's devices.
 */
void find_start_index(int device_id, int curr_query, int start_index, std::vector<std::vector<int>>& thread_start_idx,
                        int thread_id, const LocalPartitions& device_partitions) {
    for (int i = start_index; i >= 0; i--) {
        if (std::get<0>(device_partitions[device_id])[i].get_query() != curr_query) {
            thread_start_idx[device_id][thread_id] = i + 1;
            break;
        }
        else if (i == 0) {
            thread_start_idx[device_id][thread_id] = 0;
        }
    }
}

/**
 * @brief Remove unused threads from the current partition and assign them to
 * the next partition, if possible.
 *
 * @param device_id The ID of the device.
 * @param unused_threads The number of unused threads.
 * @param config Processing configuration.
 * @param thread_start_idx The start index of each thread.
 */
void reassign_unused_threads(int device_id, int unused_threads, const ProcessingConfig& config,
                             std::vector<std::vector<int>>& thread_start_idx) {
    if (unused_threads > 0) {
        thread_start_idx[device_id].resize(thread_start_idx[device_id].size() - unused_threads);
        // Add the removed threads to the next partition.
        if (device_id < config.worker_count - 1) {
            thread_start_idx[device_id + 1].resize(thread_start_idx[device_id + 1].size() + unused_threads, -1);
        }
    }
}

/**
 * @brief Determine the starting query index for each thread.
 *
 * @param device_id The ID of the device.
 * @param threads_per_device The number of threads assigned to the device.
 * @param thread_start_idx The start index of each thread.
 * @param device_partitions The dataset assigned to each of the node's devices.
 * @param stride The number of queries to be processed by each thread.
 * @return int The number of unused threads.
 */
int determine_start_indices(int device_id, int threads_per_device, std::vector<std::vector<int>>& thread_start_idx,
                            const LocalPartitions& device_partitions, int stride) {
    int unused_threads = 0;

    for (int thread_id = 0; thread_id < threads_per_device; ++thread_id) {
        int curr_query = std::get<0>(device_partitions[device_id])[thread_id * stride].get_query();
        int start_index = thread_id * stride;

        // Check if the current query has been assigned to a previous thread.
        // If so, assign the next query to the current thread.
        if (thread_id - 1 > 0) { // Check if there are any threads before the current thread.
            int prev_query = std::get<0>(device_partitions[device_id])[thread_start_idx[device_id][thread_id - 1]].get_query();
            // If the current query has already been assigned, then find the
            // next available query in the dataset.
            if (curr_query == prev_query) {
                // If no query is found that is not assigned to a previous
                // thread, increase the number of unused threads. These will be
                // reassigned later.
                unused_threads += find_next_non_duplicate_query(device_id, curr_query, start_index,
                                                                device_partitions, thread_start_idx, thread_id);
            }
        }
        find_start_index(device_id, curr_query, start_index, thread_start_idx, thread_id, device_partitions);
    }
    return unused_threads;
}

/**
 * @brief Calculate the total number of queries to be processed.
 *
 * @param device_partitions The dataset assigned to each of the node's devices.
 * @return int The total number of queries to be processed.
 */
int calculate_total_queries(const LocalPartitions& device_partitions) {
    return std::accumulate(device_partitions.begin(), device_partitions.end(), 0,
        [](int sum, const Partition& partition) {
            return sum + std::get<0>(partition).size();
        }
    );
}

/**
 * @brief Allocate CPU threads to each device based on the number of assigned
 * queries to the device.
 *
 * @param config Processing configuration.
 * @param thread_start_idx The start index of each thread.
 * @param device_partitions The dataset assigned to each of the node's devices.
 * @param total_queries The total number of queries to be processed.
 */
void allocate_threads_to_devices(const ProcessingConfig& config, std::vector<std::vector<int>>& thread_start_idx,
                                 const LocalPartitions& device_partitions, int total_queries) {
    int available_threads = config.thread_count;
    int device_id = 0; // Start thread assignment with device 0.

    while (available_threads > 0) {
        int partition_query_count = std::get<0>(device_partitions[device_id]).size();
        float ratio = static_cast<float>(partition_query_count) / total_queries;
        int threads_per_device = std::round(ratio * config.thread_count);

        // Ensure that each device always gets at least one thread and that the
        // assigned threads do not exceed the available threads.
        if (thread_start_idx[device_id].size() < 1 || thread_start_idx[device_id].size() < threads_per_device) {
            thread_start_idx[device_id].push_back(-1);
            --available_threads;
        }
        device_id = (device_id + 1) % config.worker_count; // Select next device.
    }
}

/**
 * @brief Determine the number of queries per thread for each device.
 *
 * @param config Processing configuration.
 * @param thread_start_idx The start index of each thread.
 * @param device_partitions The dataset assigned to each of the node's devices.
 */
void calculate_queries_per_thread(const ProcessingConfig& config, std::vector<std::vector<int>>& thread_start_idx,
                                  const LocalPartitions& device_partitions) {
    for (int device_id = 0; device_id < config.worker_count; ++device_id) {
        int partition_query_count = std::get<0>(device_partitions[device_id]).size();
        int threads_per_device = thread_start_idx[device_id].size();
        int stride = partition_query_count / threads_per_device; // Approximate number of queries per thread.

        // Use only a single thread if there are not enough queries to
        // distribute over all threads.
        if (partition_query_count < threads_per_device) {
            thread_start_idx[device_id].resize(1);
            thread_start_idx[device_id][0] = 0;
        } else {
            int unused_threads = determine_start_indices(device_id, threads_per_device, thread_start_idx,
                                                         device_partitions, stride);
            reassign_unused_threads(device_id, unused_threads, config, thread_start_idx);
        }
    }
}


//---------------------------------------------------------------------------//
// Allocate memory.                                                          //
//---------------------------------------------------------------------------//

/**
 * @brief Display memory usage.
 *
 * @param hostname The hostname of the node.
 * @param device_id The ID of the device.
 * @param config Processing configuration.
 * @param fmem_dev The memory in use and total memory on each device.
 * @param mem_use The amount of memory currently in use.
 */
void show_memory_usage(const char* hostname, int device_id, const ProcessingConfig& config, size_t*& fmem_dev, size_t mem_use) {
    int device = (GPU_EXECUTION(config.exec_mode)) ? device_id : -1;
    float expected_mem_usage = fmem_dev[device_id * 2] / 1e6;
    float total_mem_capacity = fmem_dev[device_id * 2 + 1] / 1e6;
    int expected_mem_percent = static_cast<int>(expected_mem_usage / total_mem_capacity * 100);
    float measured_mem_usage = mem_use / 1e6;
    int measured_mem_percent = static_cast<int>(measured_mem_usage / total_mem_capacity * 100);

    std::ostringstream output;
    output << "[" << hostname << ", " << device << "], memory expected = " << expected_mem_usage << "/" << total_mem_capacity << " MB (" << expected_mem_percent << "%), measured = " << measured_mem_usage << "/" << total_mem_capacity << " MB (" << measured_mem_percent << "%)";
    std::cout << output.str() << std::endl;
}

/**
 * @brief Initialize model parameters memory.
 *
 * @param cm_host The host-side click model.
 * @param device_id The ID of the device.
 * @param device_partition The dataset assigned to the device.
 * @param fmem_dev The memory in use and total memory on each device.
 * @param gpu Whether to initialize the parameters on the device or host.
 */
void init_model_parameters(ClickModel_Hst*& cm_host, int device_id, Partition& device_partition, size_t* fmem_dev, bool gpu) {
    cm_host->init_parameters(device_partition, fmem_dev[device_id * 2 + 1] - fmem_dev[device_id * 2], gpu);
    Communicate::error_check();
}

/**
 * @brief Allocate dataset memory on device.
 *
 * @param device_id The ID of the device.
 * @param dataset_dev The device-side dataset.
 * @param dataset_dev_tmp The reformatted host-side dataset.
 * @param dataset_size The size of the dataset.
 */
void allocate_device_dataset_memory(int device_id, SearchResult_Dev*& dataset_dev, std::vector<SearchResult_Dev>& dataset_dev_tmp, double dataset_size) {
    CUDA_CHECK(cudaMalloc(&dataset_dev, dataset_size));
    CUDA_CHECK(cudaMemcpy(dataset_dev, dataset_dev_tmp.data(),
                          dataset_size, cudaMemcpyHostToDevice));
    dataset_dev_tmp.clear();
}

/**
 * @brief Check whether the dataset size does not exceed the available device
 * memory.
 *
 * @param dataset_size The size of the dataset.
 * @param fmem_dev The memory in use and total memory on each device.
 * @param device_id The ID of the device.
 * @param hostname The hostname of the node.
 */
void check_device_memory(double dataset_size, size_t* fmem_dev, int device_id, const char* hostname) {
    if (dataset_size * 1.001 > fmem_dev[device_id * 2 + 1]) {
        Communicate::error_check("[" + std::string(hostname) + "] \033[12;31mError\033[0m: Insufficient GPU memory!\n\tAllocating dataset requires an additional " + std::to_string((dataset_size - fmem_dev[device_id * 2 + 1]) / 1e6) + " MB of GPU memory.");
    }
}

/**
 * @brief Allocate memory on the device.
 *
 * @param device_id The ID of the device.
 * @param dataset_dev The device-side dataset.
 * @param fmem_dev The memory in use and total memory on each device.
 * @param config Processing configuration.
 * @param device_partition The dataset assigned to the device.
 * @param cm_host The host-side click model.
 * @param hostname The hostname of the node.
 */
void allocate_memory_on_device(int device_id, SearchResult_Dev*& dataset_dev, size_t* fmem_dev, const ProcessingConfig& config, Partition& device_partition, ClickModel_Hst*& cm_host, const char* hostname) {
    size_t fmem, tmem, fmem_new, tmem_new;

    // Retrieve available memory in bytes.
    get_device_memory(device_id, fmem, tmem, 1);
    fmem_dev[device_id * 2] = 0; // Memory in use.
    fmem_dev[device_id * 2 + 1] = fmem; // Total available memory.

    // Convert the host-side dataset to a smaller device-side dataset.
    std::vector<SearchResult_Dev> dataset_dev_tmp;
    convert_to_device(std::get<0>(device_partition), dataset_dev_tmp);

    // Check whether the current device has enough free memory available.
    double dataset_size = dataset_dev_tmp.size() * sizeof(SearchResult_Dev);
    check_device_memory(dataset_size, fmem_dev, device_id, hostname);

    // Allocate memory for the dataset on the current device.
    allocate_device_dataset_memory(device_id, dataset_dev, dataset_dev_tmp, dataset_size);
    fmem_dev[device_id * 2] += dataset_size;

    // Allocate memory for the query dependent parameters on both the current device and host.
    init_model_parameters(cm_host, device_id, device_partition, fmem_dev, true);
    fmem_dev[device_id * 2] += cm_host->get_memory_usage();

    // Show memory usage.
    get_device_memory(device_id, fmem_new, tmem_new, 1);

    show_memory_usage(hostname, device_id, config, fmem_dev, fmem - fmem_new);
}

/**
 * @brief Allocate memory on the host.
 *
 * @param device_id The ID of the device.
 * @param fmem_dev The memory in use and total memory on each device.
 * @param config Processing configuration.
 * @param host_partition The dataset assigned to the host.
 * @param cm_host The host-side click model.
 * @param hostname The hostname of the node.
 */
void allocate_memory_on_host(int device_id, size_t* fmem_dev, const ProcessingConfig& config, Partition& host_partition, ClickModel_Hst* cm_host, const char* hostname) {
    size_t fmem, tmem, fmem_new, tmem_new;

    // Retrieve available memory in bytes.
    get_host_memory(fmem, tmem, 1);
    fmem_dev[device_id * 2] = 0; // Memory in use.
    fmem_dev[device_id * 2 + 1] = fmem; // Total available memory.

    // Dataset size does not need to be checked, since the dataset has
    // already been allocated on the host.

    // Allocate memory for the query dependent parameters on both the current device and host.
    init_model_parameters(cm_host, device_id, host_partition, fmem_dev, false);
    fmem_dev[device_id * 2] += cm_host->get_memory_usage();

    // Show memory usage.
    get_host_memory(fmem_new, tmem_new, 1);

    show_memory_usage(hostname, device_id, config, fmem_dev, fmem - fmem_new);
}


/**
 * @brief Execute the EM algorithm in parallel.
 *
 * @param config Processing configuration.
 * @param device_partitions The dataset assigned to each of the node's devices.
 */
 void em_parallel(
    const ProcessingConfig& config,
    LocalPartitions& device_partitions
) {
    Timer timer;
    ConditionalStream RCOUT(config, ROOT_RANK, std::cout);
    ConditionalStream RCERR(config, ROOT_RANK, std::cerr);

    RCOUT << "\nExpectation Maximization (EM) in parallel ..." << std::endl;

    //-----------------------------------------------------------------------//
    // Initate host-side click model.                                        //
    //-----------------------------------------------------------------------//

    // Initiate a host-side click model for each device or for the host.
    ClickModel_Hst* cm_hosts[config.worker_count];
    for (int unit = 0; unit < config.worker_count; unit++) {
        // Initialize the click model.
        cm_hosts[unit] = create_cm_host(config.model_type);
        // Print a confirmation message on the first device of the root node.
        if (config.node_id == ROOT_RANK && unit == 0) {
            cm_hosts[unit]->say_hello();
        }
    }

    //-----------------------------------------------------------------------//
    // Assign queries to CPU threads.                                        //
    //-----------------------------------------------------------------------//

    std::vector<std::vector<int>> thread_start_idx(config.worker_count);

    if (config.exec_mode != 0) {
        allocate_threads_to_devices(config, thread_start_idx, device_partitions,
                                    calculate_total_queries(device_partitions));
        calculate_queries_per_thread(config, thread_start_idx, device_partitions);
    }

    //-----------------------------------------------------------------------//
    // Allocate memory.                                                      //
    //-----------------------------------------------------------------------//

    timer.start("h2d_init");

    // Allocate memory on the device.
    SearchResult_Dev* dataset_dev[config.device_count];
    size_t fmem_dev[config.worker_count * 2];
    for (int device_id = 0; device_id < config.worker_count; device_id++) {
        if (GPU_EXECUTION(config.exec_mode)) {
            CUDA_CHECK(cudaSetDevice(device_id));
            allocate_memory_on_device(device_id, dataset_dev[device_id], fmem_dev, config, device_partitions[device_id], cm_hosts[device_id], config.host_name);
        } else {
            allocate_memory_on_host(device_id, fmem_dev, config, device_partitions[device_id], cm_hosts[device_id], config.host_name);
        }
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

        std::cout << "[" << config.host_name << ", " << did << "] kernel dimensions = <<<" << kernel_dims[did * 2] << ", " << kernel_dims[did * 2 + 1] << ">>>" << std::endl;
    }

    RCOUT << "\nStarting " << config.iterations << " EM parameter estimation iterations..." << std::endl;

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
            for (int unit = 0; unit < config.worker_count; unit++) {
                cm_hosts[unit]->process_session(std::get<0>(device_partitions[unit]), thread_start_idx[unit]);
            }
        }

        timer.lap("EM computation", false);


        //-------------------------------------------------------------------//
        // Wipe previous parameter results.                                  //
        //-------------------------------------------------------------------//

        for (int device_id = 0; device_id < config.worker_count; device_id++) {
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
            for (int unit = 0; unit < config.worker_count; unit++) {
                cm_hosts[unit]->update_parameters(std::get<0>(device_partitions[unit]), thread_start_idx[unit]);
            }
        }

        timer.lap("EM update", false);


        //-------------------------------------------------------------------//
        // Synchronize parameters across the nodes and devices.              //
        //-------------------------------------------------------------------//

        timer.start("EM synchronization");
        timer.start("d2h");

        std::vector<std::vector<std::vector<Param>>> public_parameters(config.worker_count); // Device ID -> Parameter type -> Parameters.

        // Retrieve all types of public parameters from each device.
        for (int device_id = 0; device_id < config.worker_count; device_id++) {
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
        Communicate::exchange_parameters(network_parameters, public_parameters[0]);

        // Sychronize the public parameters received from other nodes.
        Communicate::sync_parameters(network_parameters);

        // Move all types of synchronized public parameters back to each device.
        for (int device_id = 0; device_id < config.worker_count; device_id++) {
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
        int itr_len = std::to_string(config.iterations).length() + 1;
        RCOUT << "Itr: " << std::left << std::setw(itr_len) << itr
              << " Itr-time: " << std::left << std::setw(11) << timer.lap("EM iteration")
              << " Itr-EM_COMP: " << std::left << std::setw(11) << timer.elapsed("EM computation")
              << " Itr-EM_UPDATE: " << std::left << std::setw(11) << timer.elapsed("EM update")
              << " Itr-Sync: " << std::left << std::setw(12) << timer.elapsed("EM synchronization") << std::endl;
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
    for (int device_id = 0; device_id < config.worker_count; device_id++) {
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
    Communicate::gather_evaluations(llh_device, ppl_device, config.devices_per_node);

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

    if (!config.output_path.empty()) {
        RCOUT << "\nWriting output to file..." << std::endl;

        std::pair<std::vector<std::string>, std::vector<std::string>> headers;
        std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> parameters[config.worker_count];
        for (int device_id = 0; device_id < config.worker_count; device_id++) {
            cm_hosts[device_id]->get_parameter_information(headers, parameters[device_id]);
        }
        Communicate::output_parameters(config.worker_count, config.output_path, device_partitions, headers, parameters);
    }


    //-----------------------------------------------------------------------//
    // Show metrics.                                                         //
    //-----------------------------------------------------------------------//

    // Show metrics on the root node.
    if (config.node_id == ROOT_RANK) {
        // Compute the total log-likelihood.
        float total_llh_sum = 0.0;
        float total_llh_sessions = 0.0;
        std::for_each(llh_device.begin(), llh_device.end(), [&] (std::pair<const int, std::array<float, 2>>& llh_task) {
            total_llh_sum += llh_task.second[0];
            total_llh_sessions += llh_task.second[1];
        });

        RCOUT << "\nTotal Log likelihood is: " << total_llh_sum / total_llh_sessions << std::endl;

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
            RCOUT << "Perplexity at rank " << i << " is: " << ppl_per_rank[i] << std::endl;
        }
        ppl_value = std::accumulate(ppl_per_rank.begin(), ppl_per_rank.end(), 0.0) / 10.0;
        RCOUT << "Perplexity is: " << ppl_value << std::endl;

        // Show the timing measurements of the EM algorithm.
        if (GPU_EXECUTION(config.exec_mode)) {
            RCOUT << "\nHost to Device dataset transfer time: " << timer.elapsed("h2d_init")
                      << "\nAverage Host to Device parameter transfer time: " << timer.avg("h2d") / 2
                      << "\nAverage Device to Host parameter transfer time: " << timer.avg("d2h") << std::endl;
        }
        RCOUT << "\nAverage time per iteration: " << timer.avg("EM iteration")
                  << "\nAverage time per computation in each iteration: " << timer.avg("EM computation")
                  << "\nAverage time per update in each iteration: " << timer.avg("EM update")
                  << "\nAverage time per synchronization in each iteration: " << timer.avg("EM synchronization")
                  << "\nTotal time of training: " << timer.total("EM iteration")
                  << "\nEvaluation time: " << timer.elapsed("EM evaluation") << std::endl;
    }

    // Destroy all allocations on all available devices as part of the shutdown
    // procedure.
    for (int device_id = 0; device_id < (GPU_EXECUTION(config.exec_mode) ? config.device_count : 0); device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaDeviceReset());
    }
}