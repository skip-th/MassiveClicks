/** PBM click model.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * pbm.cu:
 *  - Defines the functions specific to creating a PBM CM.
 */

#include "pbm.cuh"


//---------------------------------------------------------------------------//
// Host-side PBM click model functions.                                      //
//---------------------------------------------------------------------------//

HST PBM_Hst::PBM_Hst() = default;

/**
 * @brief Constructs a PBM click model object for the host.
 *
 * @param pbm
 * @returns PBM_Hst The PBM click model object.
 */
HST PBM_Hst::PBM_Hst(PBM_Hst const &pbm) {
}

/**
 * @brief Creates a new PBM click model object.
 *
 * @return PBM_Hst* The PBM click model object.
 */
HST PBM_Hst* PBM_Hst::clone() {
    return new PBM_Hst(*this);
}

/**
 * @brief Print a message.
 */
HST void PBM_Hst::say_hello() {
    std::cout << "Host-side PBM says hello!" << std::endl;
}

/**
 * @brief Get the amount of device memory allocated to this click model.
 *
 * @return size_t The used memory.
 */
HST size_t PBM_Hst::get_memory_usage(void) {
    return this->cm_memory_usage;
}

/**
 * @brief Get the expected amount of memory the click model will need to store
 * the current parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @return size_t The worst-case parameter memory footprint.
 */
HST size_t PBM_Hst::compute_memory_footprint(int n_queries, int n_qd) {
    std::pair<int, int> n_attractiveness = this->get_n_attr_params(n_queries, n_qd);
    std::pair<int, int> n_examination = this->get_n_exam_params(n_queries, n_qd);

    return (n_attractiveness.first + n_attractiveness.second +
            n_examination.first + n_examination.second) * sizeof(Param);
}

/**
 * @brief Get the number of original and temporary attractiveness parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return std::pair<int,int> The number of original and temporary attractiveness
 * parameters.
 */
HST std::pair<int,int> PBM_Hst::get_n_attr_params(int n_queries, int n_qd) {
    return std::make_pair(n_qd,                         // # original
                          n_queries * MAX_SERP_LENGTH); // # temporary
}

/**
 * @brief Get the number of original and temporary examination parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return std::pair<int,int> The number of original and temporary examination
 * parameters.
 */
HST std::pair<int, int> PBM_Hst::get_n_exam_params(int n_queries, int n_qd) {
    return std::make_pair(MAX_SERP_LENGTH,              // # original
                          n_queries * MAX_SERP_LENGTH); // # temporary
}

/**
 * @brief Allocate device-side memory for the attractiveness parameters.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void PBM_Hst::init_attractiveness_parameters(const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition, const size_t fmem) {
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Compute the storage space required to store the parameters.
    std::pair<int, int> n_attractiveness = this->get_n_attr_params(std::get<0>(partition).size(), std::get<2>(partition));
    this->n_attr_dev = n_attractiveness.first;
    this->n_tmp_attr_dev = n_attractiveness.second;
    // Store the number of allocated bytes.
    this->cm_memory_usage += this->n_attr_dev * sizeof(Param) + this->n_tmp_attr_dev * sizeof(Param);
    // Check if the new parameters will fit in GPU memory using a 0.1% error margin.
    if (this->cm_memory_usage * 1.001 > fmem) {
        std::cout << "Error: Insufficient GPU memory!\n" <<
        "\tAllocating attractiveness parameters requires an additional " <<
        (this->cm_memory_usage - fmem) / 1e6 << " MB of GPU memory." << std::endl;
        mpi_abort(-1);
    }

    // Allocate memory for the attractiveness parameters on the device.
    this->attractiveness_parameters.resize(this->n_attr_dev, default_parameter);
    CUDA_CHECK(cudaMalloc(&this->attr_param_dptr, this->n_attr_dev * sizeof(Param)));
    CUDA_CHECK(cudaMemcpy(this->attr_param_dptr, this->attractiveness_parameters.data(),
                          this->n_attr_dev * sizeof(Param), cudaMemcpyHostToDevice));

    // Allocate memory for the temporary attractiveness parameters on the device.
    // These values are replaced at the start of each iteration, which means
    // they don't need to be initialized with a CUDA memory copy.
    this->tmp_attractiveness_parameters.resize(this->n_tmp_attr_dev);
    CUDA_CHECK(cudaMalloc(&this->tmp_attr_param_dptr, this->n_tmp_attr_dev * sizeof(Param)));
}

/**
 * @brief Allocate device-side memory for the examination parameters.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void PBM_Hst::init_examination_parameters(const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition, const size_t fmem) {
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Compute the storage space required to store the parameters.
    std::pair<int, int> n_examination = this->get_n_exam_params(std::get<0>(partition).size(), std::get<2>(partition));
    this->n_exams_dev = n_examination.first;
    this->n_tmp_exams_dev = n_examination.second;
    // Store the number of allocated bytes.
    this->cm_memory_usage += this->n_exams_dev * sizeof(Param) + this->n_tmp_exams_dev * sizeof(Param);
    // Check if the new parameters will fit in GPU memory using a 0.1% error margin.
    if (this->cm_memory_usage * 1.001 > fmem) {
        std::cout << "Error: Insufficient GPU memory!\n" <<
        "\tAllocating examination parameters requires an additional " <<
        (this->cm_memory_usage - fmem) / 1e6 << " MB of GPU memory." << std::endl;
        mpi_abort(-1);
    }

    // Allocate memory for the examination parameters on the device.
    this->examination_parameters.resize(this->n_exams_dev, default_parameter);
    CUDA_CHECK(cudaMalloc(&this->exam_param_dptr, this->n_exams_dev * sizeof(Param)));
    CUDA_CHECK(cudaMemcpy(this->exam_param_dptr, this->examination_parameters.data(),
                          this->n_exams_dev * sizeof(Param), cudaMemcpyHostToDevice));

    // Allocate memory for the temporary examination parameters on the device.
    // These values are replaced at the start of each iteration, which means
    // they don't need to be initialized with a CUDA memory copy.
    this->tmp_examination_parameters.resize(this->n_tmp_exams_dev);
    CUDA_CHECK(cudaMalloc(&this->tmp_exam_param_dptr, this->n_tmp_exams_dev * sizeof(Param)));
}

/**
 * @brief Allocate device-side memory for the attractiveness and examination
 * parameters of the click model.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void PBM_Hst::init_parameters(const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition, const size_t fmem) {
    this->init_attractiveness_parameters(partition, fmem);
    this->init_examination_parameters(partition, fmem);
}

/**
 * @brief Get the references to the allocated device-side memory.
 *
 * @param param_refs An array containing the references to the device-side
 * parameters in memory.
 * @param param_sizes The size of each of the memory allocations on the device.
 */
HST void PBM_Hst::get_device_references(Param**& param_refs, int*& param_sizes) {
    int n_references = 4;

    // Create a temporary array to store the device references.
    Param* tmp_param_refs_array[n_references];
    tmp_param_refs_array[0] = this->attr_param_dptr;
    tmp_param_refs_array[1] = this->tmp_attr_param_dptr;
    tmp_param_refs_array[2] = this->exam_param_dptr;
    tmp_param_refs_array[3] = this->tmp_exam_param_dptr;

    // Allocate space for the device references.
    CUDA_CHECK(cudaMalloc(&param_refs, n_references * sizeof(Param*)));
    CUDA_CHECK(cudaMemcpy(param_refs, tmp_param_refs_array,
                          n_references * sizeof(Param*), cudaMemcpyHostToDevice));

    int tmp_param_sizes_array[n_references];
    tmp_param_sizes_array[0] = this->n_attr_dev;
    tmp_param_sizes_array[1] = this->n_tmp_attr_dev;
    tmp_param_sizes_array[2] = this->n_exams_dev;
    tmp_param_sizes_array[3] = this->n_tmp_exams_dev;

    // Allocate space for the device references.
    CUDA_CHECK(cudaMalloc(&param_sizes, n_references * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(param_sizes, tmp_param_sizes_array,
                          n_references * sizeof(int), cudaMemcpyHostToDevice));

    // Keep track of the pointers to the allocated device-side memory.
    this->param_refs = param_refs;
    this->param_sizes = param_sizes;
    this->cm_memory_usage += n_references * sizeof(Param*) + n_references * sizeof(int);
}

/**
 * @brief Update the global parameter values with the temporarily stored new
 * local parameter values on each thread.
 *
 * @param gridSize The size of kernel blocks on the GPU.
 * @param blockSize The number of kernel threads per block on the GPU.
 * @param partition The dataset allocated on the GPU.
 * @param dataset_size The size of the allocated dataset.
 */
HST void PBM_Hst::update_parameters(int& gridSize, int& blockSize, SERP_Dev*& partition, int& dataset_size) {
    Kernel::update<<<gridSize, blockSize>>>(partition, dataset_size);
}

// struct thread_data {
//     int thread_id;
//     int start_idx;
//     int stop_idx;
//     std::vector<SERP_Hst>* partition;
// };

// HST void* PBM_Hst::update_examination_parameters(void* data) {
//     thread_data* ptr = (thread_data*) data;
//     int thread_id = ptr->thread_id;
//     int start_idx = ptr->start_idx;
//     int stop_idx = ptr->stop_idx;
//     std::vector<SERP_Hst>* partition = ptr->partition;

//     // Array to store the final parameter values, initialized at 0.
//     Param* public_sum = (Param*) calloc(MAX_SERP_LENGTH, sizeof(Param));

//     // Sum the public parameters.
//     for (int query_index = start_idx; query_index < stop_idx; query_index++) {
//         for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
//             public_sum[rank] += this->tmp_examination_parameters[rank * partition->size() + query_index];
//         }
//     }

//     // Exit the pthread and return the summed public parameters.
//     pthread_exit(public_sum);
// }

// HST void* PBM_Hst::update_attractiveness_parameters(void* data) {
//     thread_data* ptr = (thread_data*) data;
//     int thread_id = ptr->thread_id;
//     int start_idx = ptr->start_idx;
//     int stop_idx = ptr->stop_idx;
//     std::vector<SERP_Hst>* partition = ptr->partition;

//     // Sum the private parameters atomically.
//     for (int query_index = start_idx; query_index < stop_idx; query_index++) {
//         for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
//             SearchResult_Hst sr = (*partition)[query_index][rank];
//             this->attractiveness_parameters[sr.get_param_index()].add_to_values(
//                 this->tmp_attractiveness_parameters[rank * partition->size() + query_index].numerator_val(),
//                 1.f);
//         }
//     }

//     // Exit the pthread.
//     pthread_exit(NULL);
// }

// HST void PBM_Hst::update_parameters_on_host(const std::vector<int>& thread_start_idx, std::vector<SERP_Hst>& partition) {
//     // Retrieve the intermediate parameter values.
//     CUDA_CHECK(cudaMemcpy(this->tmp_examination_parameters.data(), this->tmp_exam_param_dptr, this->n_tmp_exams_dev * sizeof(Param), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(this->tmp_attractiveness_parameters.data(), this->tmp_attr_param_dptr, this->n_tmp_attr_dev * sizeof(Param), cudaMemcpyDeviceToHost));

//     int n_threads = thread_start_idx.size();
//     pthread_t threads[n_threads];
//     struct thread_data data[n_threads];

//     // Reset parameters on the host.
//     Param default_parameter;
//     default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);
//     std::fill(this->examination_parameters.begin(), this->examination_parameters.end(), default_parameter);
//     std::fill(this->attractiveness_parameters.begin(), this->attractiveness_parameters.end(), default_parameter);

//     // Initialize examination parameter update threads.
//     std::vector<Param*> public_results(n_threads);
//     for (int i = 0; i < n_threads; i++) {
//         data[i].thread_id = i;
//         data[i].start_idx = thread_start_idx[i];
//         if (i == n_threads - 1) {
//             data[i].stop_idx = partition.size();
//         } else {
//             data[i].stop_idx = thread_start_idx[i + 1];
//         }
//         data[i].partition = &partition;

//         if (pthread_create(&threads[i], NULL, update_ex_init, (void*) &data[i])) {
//             perror("Error: failed to create examination update pthread.");
//             mpi_abort(-1);
//         }
//     }

//     // Wait for all threads to finish and gather the results.
//     for (int i = 0; i < n_threads; i++) {
//         if (pthread_join(threads[i], (void**) &public_results[i])) {
//             perror("Error: failed to join examination update pthread.");
//             mpi_abort(-1);
//         }
//     }

//     // Sum the results of each thread.
//     for (int thread_index = 0; thread_index < n_threads; thread_index++) {
//         for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
//             this->examination_parameters[rank] += public_results[thread_index][rank];
//         }
//     }

//     // Initialize attractiveness parameter update threads.
//     for (int i = 0; i < n_threads; i++) {
//         data[i].thread_id = i;
//         data[i].start_idx = thread_start_idx[i];
//         if (i == n_threads - 1) {
//             data[i].stop_idx = partition.size();
//         } else {
//             data[i].stop_idx = thread_start_idx[i + 1];
//         }
//         data[i].partition = &partition;

//         if (pthread_create(&threads[i], NULL, update_attr_init, (void*) &data[i])) {
//             perror("Error: failed to create attractiveness update pthread.");
//             mpi_abort(-1);
//         }
//     }

//     // Wait for all threads to finish.
//     for (int i = 0; i < n_threads; i++) {
//         if (pthread_join(threads[i], NULL)) {
//             perror("Error: failed to join attractiveness update pthread.");
//             mpi_abort(-1);
//         }
//     }

//     // Move the private parameters back to the GPU (the public parameters will
//     // be moved back later).
//     this->transfer_parameters(PRIVATE, H2D);
// }

HST void PBM_Hst::update_parameters_on_host(const std::vector<int>& thread_start_idx, std::vector<SERP_Hst>& partition) {
    // Retrieve the intermediate parameter values.
    this->transfer_parameters(PUBLIC, D2H);
    this->transfer_parameters(PRIVATE, D2H);
    CUDA_CHECK(cudaMemcpy(this->tmp_examination_parameters.data(), this->tmp_exam_param_dptr, this->n_tmp_exams_dev * sizeof(Param), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(this->tmp_attractiveness_parameters.data(), this->tmp_attr_param_dptr, this->n_tmp_attr_dev * sizeof(Param), cudaMemcpyDeviceToHost));

    // Update attractiveness parameters.
    for (int query_index = 0; query_index < partition.size(); query_index++) {
        SERP_Hst query_session = partition[query_index];
        for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
            SearchResult_Hst sr = query_session[rank];
            this->attractiveness_parameters[sr.get_param_index()].add_to_values(
                this->tmp_attractiveness_parameters[rank * partition.size() + query_index].numerator_val(),
                1.f);
        }
    }

    // Update examination parameters.
    for (int query_index = 0; query_index < partition.size(); query_index++) {
        for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
            Param* tmp_param = &this->tmp_examination_parameters[rank * partition.size() + query_index];
            this->examination_parameters[rank].add_to_values(tmp_param->numerator_val(),
                                                             tmp_param->denominator_val());
        }
    }

    // Move the private parameters back to the GPU (the public parameters will
    // be moved back later).
    this->transfer_parameters(PUBLIC, H2D);
    this->transfer_parameters(PRIVATE, H2D);
}

/**
 * @brief Reset the original parameter values to zero so the previous parameter
 * values won't affect the next result twice.
 *
 * Further explanation; The first time it would affect the result would be when
 * retrieving their values in the training kernel, and the (unnecessary) second
 * time would be when adding the values to the original parameter containers.
 * The second time would still give a valid result but would slow down the
 * converging of the parameters.
 */
HST void PBM_Hst::reset_parameters(void) {
    // Create a parameter initialized at 0.
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Create an array of the right proportions with the empty parameters.
    std::vector<Param> cleared_examination_parameters(this->n_exams_dev, default_parameter);
    std::vector<Param> cleared_attractiveness_parameters(this->n_attr_dev, default_parameter);

    // Copy the cleared array to the device.
    CUDA_CHECK(cudaMemcpy(this->exam_param_dptr, cleared_examination_parameters.data(), this->n_exams_dev * sizeof(Param), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(this->attr_param_dptr, cleared_attractiveness_parameters.data(), this->n_attr_dev * sizeof(Param), cudaMemcpyHostToDevice));
}

/**
 * @brief Transfers parameters of a given type either from the device to the
 * host, or the otherway around.
 *
 * @param parameter_type The type of parameter that will be transfered.
 * (PUBLIC, PRIVATE, or ALL).
 * @param transfer_direction The direction in which the transfer will happen.
 * (H2D or D2H).
 */
HST void PBM_Hst::transfer_parameters(int parameter_type, int transfer_direction) {
    // Public parameters.
    if (parameter_type == PUBLIC || parameter_type == ALL) {
        if (transfer_direction == D2H) { // Transfer from device to host.
            // Retrieve the examination parameters from the device.
            CUDA_CHECK(cudaMemcpy(this->examination_parameters.data(), this->exam_param_dptr, this->n_exams_dev * sizeof(Param), cudaMemcpyDeviceToHost));
        }
        else if (transfer_direction == H2D) { // Transfer from host to device.
            // Send the examination parameters to the device.
            CUDA_CHECK(cudaMemcpy(this->exam_param_dptr, this->examination_parameters.data(), this->n_exams_dev * sizeof(Param), cudaMemcpyHostToDevice));
        }
    }

    // Private parameters.
    if (parameter_type == PRIVATE || parameter_type == ALL) {
        if (transfer_direction == D2H) { // Transfer from device to host.
            // Retrieve the attractiveness parameters from the device.
            CUDA_CHECK(cudaMemcpy(this->attractiveness_parameters.data(), this->attr_param_dptr, this->n_attr_dev * sizeof(Param), cudaMemcpyDeviceToHost));
        }
        else if (transfer_direction == H2D) { // Transfer from host to device.
            // Send the attractiveness parameters to the device.
            CUDA_CHECK(cudaMemcpy(this->attr_param_dptr, this->attractiveness_parameters.data(), this->n_attr_dev * sizeof(Param), cudaMemcpyHostToDevice));
        }
    }
}

/**
 * @brief Retrieve the parameters of a given type into a given array from the
 * click model.
 *
 * @param destination The array which will save the indicated parameters.
 * @param parameter_type The type of parameters which will be retrieved
 * (PUBLIC, PRIVATE, or ALL).
 */
HST void PBM_Hst::get_parameters(std::vector<std::vector<Param>>& destination, int parameter_type) {
    // Add the parameters to a generic vector which can represent  multiple
    // retrieved parameter types.
    if (parameter_type == PUBLIC) {
        destination.resize(1);
        destination[0] = this->examination_parameters;
    }
    else if (parameter_type == PRIVATE) {
        destination.resize(1);
        destination[0] = this->attractiveness_parameters;
    }
    else if (parameter_type == ALL) {
        destination.resize(2);
        destination[0] = this->examination_parameters;
        destination[1] = this->attractiveness_parameters;
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
HST void PBM_Hst::sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters) {
    for (int rank = 0; rank < parameters[0][0].size(); rank++) {
        for (int param_type = 0; param_type < parameters[0].size(); param_type++) {
            Param base = parameters[0][param_type][rank];

            // Subtract the starting values of other partitions.
            parameters[0][param_type][rank].set_values(base.numerator_val() - (parameters.size() - 1),
                                                       base.denominator_val() - 2 * (parameters.size() - 1));

            for (int device_id = 1; device_id < parameters.size(); device_id++) {
                Param ex = parameters[device_id][param_type][rank];
                parameters[0][param_type][rank].add_to_values(ex.numerator_val(),
                                                              ex.denominator_val());
            }
        }
    }
}

/**
 * @brief Set the parameters of a host-side click model equal to the given
 * given arguments.
 *
 * @param source The new parameter values.
 * @param parameter_type The type of the given parameters. (PUBLIC, PRIVATE, or
 * ALL).
 */
HST void PBM_Hst::set_parameters(std::vector<std::vector<Param>>& source, int parameter_type) {
    // Set the parameters of this click model.
    if (parameter_type == PUBLIC) {
        this->examination_parameters = source[0];
    }
    else if (parameter_type == PRIVATE) {
        this->attractiveness_parameters = source[0];
    }
    else if (parameter_type == ALL) {
        this->examination_parameters = source[0];
        this->attractiveness_parameters = source[1];
    }
}

/**
 * @brief Compute the log-likelihood of the current PBM for the given query
 * session.
 *
 * @param query_session The query session for which the log-likelihood will be
 * computed.
 * @param log_click_probs The vector which will store the log-likelihood for
 * the document at each rank in the query session.
 */
HST void PBM_Hst::get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) {
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult_Hst sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float atr{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1)
            atr = this->attractiveness_parameters[sr.get_param_index()].value();
        float ex{this->examination_parameters[rank].value()};

        // Calculate the click probability.
        float atr_mul_ex = atr * ex;

        // Calculate the log click probability.
        int click{sr.get_click()};
        if (click == 1) {
            log_click_probs.push_back(std::log(atr_mul_ex));
        }
        else {
            log_click_probs.push_back(std::log(1 - atr_mul_ex));
        }
    }
}

/**
 * @brief Compute the click probability of the current PBM for the given query
 * session.
 *
 * @param query_session The query session for which the click probability will
 * be computed.
 * @param full_click_probs The vector which will store the click probability
 * for the document at each rank in the query session.
 */
HST void PBM_Hst::get_full_click_probs(SERP_Hst& query_session, std::vector<float> &full_click_probs) {
    // Go through all ranks of the query session.
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        // Retrieve the search result at the current rank.
        SearchResult_Hst sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float atr{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1)
            atr = this->attractiveness_parameters[sr.get_param_index()].value();
        float ex{this->examination_parameters[rank].value()};

        // Calculate the click probability.
        float atr_mul_ex = atr * ex;

        // Calculate the full click probability.
        int click{sr.get_click()};
        if (click == 1) {
            full_click_probs.push_back(atr_mul_ex);
        }
        else {
            full_click_probs.push_back(1 - atr_mul_ex);
        }
    }
}

/**
 * @brief Frees the memory allocated to the parameters of this click model on
 * the GPU device.
 */
HST void PBM_Hst::destroy_parameters(void) {
    // Free origin and temporary attractiveness containers.
    CUDA_CHECK(cudaFree(this->attr_param_dptr));
    CUDA_CHECK(cudaFree(this->tmp_attr_param_dptr));

    // Free origin and temporary examination containers.
    CUDA_CHECK(cudaFree(this->exam_param_dptr));
    CUDA_CHECK(cudaFree(this->tmp_exam_param_dptr));

    // Free the device parameter references and sizes.
    CUDA_CHECK(cudaFree(this->param_refs));
    CUDA_CHECK(cudaFree(this->param_sizes));

    // Reset used device memory.
    this->cm_memory_usage = 0;
}


//---------------------------------------------------------------------------//
// Device-side PBM click model functions.                                    //
//---------------------------------------------------------------------------//

/**
 * @brief Prints a message.
 */
DEV void PBM_Dev::say_hello() {
    printf("Device-side PBM says hello!\n");
}

/**
 * @brief Creates a new PBM click model object.
 *
 * @return PBM_Dev* The PBM click model object.
 */
DEV PBM_Dev *PBM_Dev::clone() {
    return new PBM_Dev(*this);
}

DEV PBM_Dev::PBM_Dev() = default;

/**
 * @brief Constructs a PBM click model object for the device.
 *
 * @param pbm
 * @returns PBM_Dev The PBM click model object.
 */
DEV PBM_Dev::PBM_Dev(PBM_Dev const &pbm) {
}

/**
 * @brief Set the location of the memory allocated for the parameters of this
 * click model on the GPU device.
 *
 * @param parameter_ptr The pointers to the allocated memory.
 * @param parameter_sizes The size of the allocated memory.
 */
DEV void PBM_Dev::set_parameters(Param**& parameter_ptr, int* parameter_sizes) {
    this->attractiveness_parameters = parameter_ptr[0];
    this->tmp_attractiveness_parameters = parameter_ptr[1];
    this->examination_parameters = parameter_ptr[2];
    this->tmp_examination_parameters = parameter_ptr[3];

    this->n_attractiveness_parameters = parameter_sizes[0];
    this->n_tmp_attractiveness_parameters = parameter_sizes[1];
    this->n_examination_parameters = parameter_sizes[2];
    this->n_tmp_examination_parameters = parameter_sizes[3];
}

/**
 * @brief Compute a single Expectation-Maximization iteration for the PBM click
 * model, for a single query session.
 *
 * @param query_session The query session which will be used to estimate the
 * PBM parameters.
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 */
DEV void PBM_Dev::process_session(SERP_Dev& query_session, int& thread_index, int& partition_size) {
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult_Dev sr = query_session[rank];

        // Get the attractiveness and examination parameters.
        float atr{this->attractiveness_parameters[sr.get_param_index()].value()};
        float ex{this->examination_parameters[rank].value()};

        // Set the default values of the attractiveness and examination
        // parameters. These will be the parameter values in case the search
        // result document has been clicked.
        float new_numerator_atr{1};
        float new_numerator_ex{1};

        // If the search result document hasn't been clicked, then calculate
        // estimate the parameter value.
        if (sr.get_click() == 0) {
            // Calculate the current qd-pair click probability.
            float atr_ex = atr * ex;

            // Attractiveness = ((1 - gamma_{r}^{(t)}) * alpha_{qd}^{(t)}) / (1 - (gamma_{r}^{(t)} * alpha_{qd}^{(t)}))
            new_numerator_atr = (atr - atr_ex) / (1 - atr_ex);
            // Examination = ((1 - alpha_{qd}^{(t)}) * gamma_{r}^{(t)}) / (1 - (gamma_{r}^{(t)} * alpha_{qd}^{(t)}))
            new_numerator_ex = (ex - atr_ex) / (1 - atr_ex);
        }

        // Store the temporary attractiveness and examination parameters.
        this->tmp_attractiveness_parameters[rank * partition_size + thread_index].set_values(new_numerator_atr, 1);
        this->tmp_examination_parameters[rank * partition_size + thread_index].set_values(new_numerator_ex, 1);
    }
}

/**
 * @brief Update the global parameter values using the local parameter values
 * on each thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of the thread.
 * @param block_index The index of the block in which this thread exists.
 * @param parameter_type The type of parameter to update.
 * @param partition_size The size of the dataset.
 */
DEV void PBM_Dev::update_parameters(SERP_Dev& query_session, int& thread_index, int& block_index, int& partition_size) {
    this->update_examination_parameters(query_session, thread_index, block_index, partition_size);

    if (thread_index < partition_size) {
        this->update_attractiveness_parameters(query_session, thread_index, partition_size);
    }
}

/**
 * @brief Update the global examination parameters using the local examination
 * parameters of a single thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of this thread.
 * @param block_index The index of the block in which this thread exists.
 * @param partition_size The size of the dataset.
 */
DEV void PBM_Dev::update_examination_parameters(SERP_Dev& query_session, int& thread_index, int& block_index, int& partition_size) {
    SHR float numerator[BLOCK_SIZE];
    SHR float denominator[BLOCK_SIZE];

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        // Initialize shared memory for this block's parameters.
        if (thread_index < partition_size) {
            Param tmp_param = this->tmp_examination_parameters[rank * partition_size + thread_index];
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
            this->examination_parameters[rank].atomic_add_to_values(numerator[0],
                                                                    denominator[0]);
        }
    }
}

/**
 * @brief Update the global attractiveness parameters using the local
 * attractiveness parameters of a single thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of this thread.
 */
DEV void PBM_Dev::update_attractiveness_parameters(SERP_Dev& query_session, int& thread_index, int& partition_size) {
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult_Dev sr = query_session[rank];
        this->attractiveness_parameters[sr.get_param_index()].atomic_add_to_values(
            this->tmp_attractiveness_parameters[rank * partition_size + thread_index].numerator_val(),
            1.f);
    }
}
