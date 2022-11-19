/** Click model commonly used functions.
 *
 * common.cu:
 *  - Defines the functions for updating host-side parameters.
 *  -
 */

#include "common.cuh"


/**
 * @brief Allocate (device-side) memory for a set of parameters (e.g.
 * examination, attractiveness, etc.).
 *
 * @param params Pointer to the parameters to allocate space for on the device.
 * @param params_tmp Pointer to the temporary parameters to allocate space for
 * on the device.
 * @param param_dptr Pointer to the device-side parameters.
 * @param param_tmp_dptr Pointer to the device-side temporary parameters.
 * @param n_params Number of original and temporary parameters to allocate
 * space for.
 * @param n_params_dev Pointer to the number of device-side parameters.
 * @param n_params_tmp_dev Pointer to the number of device-side temporary
 * parameters.
 * @param cm_memory_usage Memory usage of the click model.
 * @param dataset The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param fmem The free memory on the device.
 * @param device Also allocate parameters on the device or not.
 */
HST void init_parameters_hst(std::vector<Param>& params, std::vector<Param>& params_tmp, Param*& param_dptr, Param*& param_tmp_dptr, std::pair<int, int> n_params, int& n_params_dev, int& n_params_tmp_dev,
                             size_t& cm_memory_usage, const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& dataset, const size_t fmem, const bool device) {
    Param default_parameter(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Compute the storage space required to store the parameters.
    n_params_dev = n_params.first;
    n_params_tmp_dev = n_params.second;
    // Store the number of allocated bytes.
    cm_memory_usage += n_params_dev * sizeof(Param) + n_params_tmp_dev * sizeof(Param);

    // Check if the new parameters will fit in GPU memory using a 0.1% error margin.
    if (cm_memory_usage * 1.001 > fmem) {
        std::cerr<< "Error: Insufficient memory!\n" <<
        "\tAllocating parameters requires an additional " <<
        (cm_memory_usage - fmem) / 1e6 << " MB of memory." << std::endl;
        mpi_abort(-1);
    }

    // Allocate memory for the parameters on the device.
    params.resize(n_params_dev, default_parameter);
    if (device) {
        CUDA_CHECK(cudaMalloc(&param_dptr, n_params_dev * sizeof(Param)));
        CUDA_CHECK(cudaMemcpy(param_dptr, params.data(),
                              n_params_dev * sizeof(Param), cudaMemcpyHostToDevice));
    }

    // Allocate memory for the temporary parameters on the device.
    // These values are replaced at the start of each iteration, which means
    // they don't need to be initialized with a CUDA memory copy.
    params_tmp.resize(n_params_tmp_dev);
    if (device) {
        CUDA_CHECK(cudaMalloc(&param_tmp_dptr, n_params_tmp_dev * sizeof(Param)));
    }
}

/**
 * @brief Reset the original parameter values to zero so the previous parameter
 * values won't affect the next result twice.
 *
 * @param params Parameters to reset.
 * @param dptr Pointer to the device-side parameters.
 * @param device Reset the parameters on the device or not.
 */
HST void reset_parameters_hst(std::vector<Param>& params, Param* dptr, bool device) {
    // Create a parameter initialized at the default value.
    Param default_parameter(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Create an array of the right proportions with the empty parameters.
    std::fill(params.begin(), params.end(), default_parameter);

    // Copy the cleared array to the device.
    if (device) {
        CUDA_CHECK(cudaMemcpy(dptr, params.data(), params.size() * sizeof(Param), cudaMemcpyHostToDevice));
    }
}

/**
 * @brief Transfers parameters of a given type either from the device to the
 * host, or the otherway around.
 *
 * @param transfer_direction The direction in which the transfer will happen.
 * (H2D or D2H).
 * @param params Parameters to transfer.
 * @param dptr Pointer to the device-side parameters.
 */
HST void transfer_parameters_hst(int transfer_direction, std::vector<Param>& params, Param* dptr) {
    if (transfer_direction == D2H) { // Transfer from device to host.
        // Retrieve the parameters from the device.
        CUDA_CHECK(cudaMemcpy(params.data(), dptr, params.size() * sizeof(Param), cudaMemcpyDeviceToHost));
    }
    else if (transfer_direction == H2D) { // Transfer from host to device.
        // Send the parameters to the device.
        CUDA_CHECK(cudaMemcpy(dptr, params.data(), params.size() * sizeof(Param), cudaMemcpyHostToDevice));
    }
}

/**
 * @brief Update parameter values unique to a query (e.g. attractiveness,
 * satisfaction) using temporary parameter values.
 *
 * @param src The (temporary) parameters used to update dst.
 * @param dst The parameter array to be updated.
 * @param dataset The training set.
 * @param thread_start_idx The dataset starting index of the range of query
 * sessions updated by each thread.
 */
HST void update_unique_parameters_hst(std::vector<Param>& src, std::vector<Param>& dst, const std::vector<SERP_Hst>& dataset,  const std::vector<int>& thread_start_idx) {
    auto update_thread = [](std::vector<Param>& src, std::vector<Param>& dst, const std::vector<SERP_Hst>& dataset, const int thread_id, const int start_idx, const int stop_idx) {
        int dataset_size = dataset.size();

        for (int query_index = start_idx; query_index < stop_idx; query_index++) {
            for (int rank = 0; rank < MAX_SERP; rank++) {
                dst[dataset[query_index][rank].get_param_index()] += src[rank * dataset_size + query_index];
            }
        }
    };

    // Create threads.
    int n_threads = thread_start_idx.size();
    std::thread threads[n_threads];

    // Launch attractiveness update threads.
    for (int tid = 0; tid < n_threads; tid++) {
        int start_idx = thread_start_idx[tid];
        int stop_idx = tid != thread_start_idx.size() - 1 ? thread_start_idx[tid + 1] : dataset.size();
        threads[tid] = std::thread(update_thread, std::ref(src), std::ref(dst), std::cref(dataset), tid, start_idx, stop_idx);
    }

    // Join attractiveness update threads.
    for (int tid = 0; tid < n_threads; tid++) {
        threads[tid].join();
    }
}

/**
 * @brief Update parameter values shared by all queries (e.g. examination,
 * continuation) using temporary parameter values.
 *
 * @param src The (temporary) parameters used to update dst.
 * @param dst The parameter array to be updated.
 * @param dataset The training set.
 * @param thread_start_idx The dataset starting index of the range of query
 * sessions updated by each thread.
 */
HST void update_shared_parameters_hst(std::vector<Param>& src, std::vector<Param>& dst, const std::vector<SERP_Hst>& dataset, const std::vector<int>& thread_start_idx) {
    auto update_thread = [](std::vector<Param>& src, std::vector<Param>& dst, const std::vector<SERP_Hst>& dataset, const int thread_id, const int start_idx, const int stop_idx, std::vector<std::mutex>& mtx) {
        int dataset_size = dataset.size();

        // Update the parameters and store the results in a local array.
        Param default_parameter(0.f, 0.f);
        std::vector<Param> dst_local(dst.size(), default_parameter);
        for (int query_index = start_idx; query_index < stop_idx; query_index++) {
            for (int rank = 0; rank < dst.size(); rank++) {
                dst_local[rank] += src[rank * dataset_size + query_index];
            }
        }

        // Update the global parameters with the local results.
        int index, offset{thread_id % (int) dst.size()};
        for (int rank = 0; rank < dst.size(); rank++) {
            index = (rank + offset) % dst.size();
            mtx[index].lock(); // Lock the mutex for this rank.
            dst[index] += dst_local[index];
            mtx[index].unlock(); // Unlock the mutex for this rank.
        }
    };

    // Create threads.
    int n_threads = thread_start_idx.size();
    std::thread threads[n_threads];

    // Launch continuation update threads.
    std::vector<std::mutex> mtx(dst.size());
    for (int tid = 0; tid < n_threads; tid++) {
        int start_idx = thread_start_idx[tid];
        int stop_idx = tid != thread_start_idx.size() - 1 ? thread_start_idx[tid + 1] : dataset.size();
        threads[tid] = std::thread(update_thread, std::ref(src), std::ref(dst), std::cref(dataset), tid, start_idx, stop_idx, std::ref(mtx));
    }

    // Join continuation update threads.
    for (int tid = 0; tid < n_threads; tid++) {
        threads[tid].join();
    }
}

/**
 * @brief Update the global parameters using the temporary parameters from a
 * single thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of this thread.
 * @param block_index The index of the block in which this thread exists.
 * @param dataset_size The size of the dataset.
 */
DEV void update_shared_parameters_dev(Param*& src, Param*& dst, int& thread_index, int& src_size, int& block_index, int& dataset_size) {
    SHR float numerator[BLOCK_SIZE];
    SHR float denominator[BLOCK_SIZE];

    for (int rank = 0; rank < src_size; rank++) {
        // Initialize shared memory for this block's parameters.
        if (thread_index < dataset_size) {
            Param tmp_param = src[rank * dataset_size + thread_index];
            numerator[block_index] = tmp_param.numerator_val();
            denominator[block_index] = tmp_param.denominator_val();
        }
        else {
            numerator[block_index] = 0.f;
            denominator[block_index] = 0.f;
        }
        // Wait for all threads to finish initializing shared memory.
        __syncthreads();

        // Perform sum reducution in shared memory using an unrolled
        // for-loop.
        if (BLOCK_SIZE >= 512) {
            if (block_index < 256) {
                numerator[block_index] += numerator[block_index + 256];
                denominator[block_index] += denominator[block_index + 256];
            }
            __syncthreads();
        }
        if (BLOCK_SIZE >= 256) {
            if (block_index < 128) {
                numerator[block_index] += numerator[block_index + 128];
                denominator[block_index] += denominator[block_index + 128];
            }
            __syncthreads();
        }
        if (BLOCK_SIZE >= 128) {
            if (block_index < 64) {
                numerator[block_index] += numerator[block_index + 64];
                denominator[block_index] += denominator[block_index + 64];
            }
            __syncthreads();
        }

        // Use an unrolled version of the reduction loop for the last 64
        // elements without explicit thread synchronization. Warp-level
        // threads (<32) are synchronized automatically.
        if (block_index < 32) {
            warp_reduce(numerator, block_index);
            warp_reduce(denominator, block_index);
        }

        // Have only the first thread of the block write the shared memory results
        // to global memory.
        if (block_index == 0) {
            dst[rank].atomic_add_to_values(numerator[0],
                                           denominator[0]);
        }
    }
}

/**
 * @brief Update the global parameters using the temporary parameters from a
 * single thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of this thread.
 * @param pidx The index of each search result's parameters to update.
 */
DEV void update_unique_parameters_dev(Param*& src, Param*& dst, int& thread_index, int& dataset_size, const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    for (int rank = 0; rank < MAX_SERP; rank++) {
        Param update = src[rank * dataset_size + thread_index];
        dst[pidx[rank * BLOCK_SIZE + threadIdx.x]].atomic_add_to_values(
            update.numerator_val(),
            update.denominator_val());
    }
}