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
 * @brief Allocate device-side memory for the attractiveness parameters.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void PBM_Hst::init_attractiveness_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, int n_devices) {
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Allocate memory for the attractiveness parameters on the device.
    this->n_attr_dev = std::get<2>(partition);
    this->attractiveness_parameters.resize(this->n_attr_dev, default_parameter);
    CUDA_CHECK(cudaMalloc(&this->attr_param_dptr, this->n_attr_dev * sizeof(Param)));
    CUDA_CHECK(cudaMemcpy(this->attr_param_dptr, this->attractiveness_parameters.data(),
                          this->n_attr_dev * sizeof(Param), cudaMemcpyHostToDevice));

    // Allocate memory for the temporary attractiveness parameters on the device.
    // These values are replaced at the start of each iteration, which means
    // they don't need to be initialized with a CUDA memory copy.
    this->n_tmp_attr_dev = std::get<0>(partition).size() * MAX_SERP_LENGTH;
    this->tmp_attractiveness_parameters.resize(this->n_tmp_attr_dev);
    CUDA_CHECK(cudaMalloc(&this->tmp_attr_param_dptr, this->n_tmp_attr_dev * sizeof(Param)));

    // Store the number of allocated bytes.
    this->cm_memory_usage += this->n_attr_dev * sizeof(Param) + this->n_tmp_attr_dev * sizeof(Param);
}

/**
 * @brief Allocate device-side memory for the examination parameters.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void PBM_Hst::init_examination_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, int n_devices) {
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Allocate memory for the examination parameters on the device.
    this->n_exams_dev = MAX_SERP_LENGTH;
    this->examination_parameters.resize(this->n_exams_dev, default_parameter);
    CUDA_CHECK(cudaMalloc(&this->exam_param_dptr, this->n_exams_dev * sizeof(Param)));
    CUDA_CHECK(cudaMemcpy(this->exam_param_dptr, this->examination_parameters.data(),
                          this->n_exams_dev * sizeof(Param), cudaMemcpyHostToDevice));

    // Allocate memory for the temporary examination parameters on the device.
    // These values are replaced at the start of each iteration, which means
    // they don't need to be initialized with a CUDA memory copy.
    this->n_tmp_exams_dev = std::get<0>(partition).size() * this->n_exams_dev;
    this->tmp_examination_parameters.resize(this->n_tmp_exams_dev);
    CUDA_CHECK(cudaMalloc(&this->tmp_exam_param_dptr, this->n_tmp_exams_dev * sizeof(Param)));

    // Store the number of allocated bytes.
    this->cm_memory_usage += this->n_exams_dev * sizeof(Param) + this->n_tmp_exams_dev * sizeof(Param);
}

/**
 * @brief Allocate device-side memory for the attractiveness and examination
 * parameters of the click model.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void PBM_Hst::init_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, int n_devices) {
    this->init_attractiveness_parameters(partition, n_devices);
    this->init_examination_parameters(partition, n_devices);
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
HST void PBM_Hst::update_parameters(int& gridSize, int& blockSize, SERP*& partition, int& dataset_size) {
    Kernel::update<<<gridSize, blockSize>>>(partition, dataset_size);

    // for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
    //     typedef thrust::device_vector<int>::iterator Iterator;
    //     strided_range<Iterator> examination(this->tmp_exam_param_dptr + rank, this->tmp_exam_param_dptr + n_tmp_exams_dev, MAX_SERP_LENGTH);
    //     thrust::reduce(examination.begin(), examination.end())
    // }
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
HST void PBM_Hst::get_log_conditional_click_probs(SERP& query_session, std::vector<float>& log_click_probs) {
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float atr{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1)
            atr = this->attractiveness_parameters[sr.get_param_index()].value();
        float ex{this->examination_parameters[sr.get_doc_rank()].value()};

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
HST void PBM_Hst::get_full_click_probs(SERP& query_session, std::vector<float> &full_click_probs) {
    // Go through all ranks of the query session.
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        // Retrieve the search result at the current rank.
        SearchResult sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float atr{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1)
            atr = this->attractiveness_parameters[sr.get_param_index()].value();
        float ex{this->examination_parameters[sr.get_doc_rank()].value()};

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
DEV void PBM_Dev::process_session(SERP& query_session, int& thread_index) {
    // int query_id = query_session.get_query();
    // printf("%d\n", query_session.get_query());
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult sr = query_session[rank];

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
        this->tmp_attractiveness_parameters[thread_index * MAX_SERP_LENGTH + rank].set_values(new_numerator_atr, 1);
        this->tmp_examination_parameters[thread_index * MAX_SERP_LENGTH + rank].set_values(new_numerator_ex, 1);
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
DEV void PBM_Dev::update_parameters(SERP& query_session, int& thread_index, int& block_index, int& partition_size) {
    this->update_examination_parameters(query_session, thread_index, block_index, partition_size);

    if (thread_index < partition_size) {
        this->update_attractiveness_parameters(query_session, thread_index);
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
DEV void PBM_Dev::update_examination_parameters(SERP& query_session, int& thread_index, int& block_index, int& partition_size) {
    // Initialize shared memory for this block's examination parameters at 0.
    SHR float block_examination_num[MAX_SERP_LENGTH];
    SHR float block_examination_denom;
    block_examination_denom = 0.f;
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        block_examination_num[rank] = 0.f;
    }
    // Wait for all threads to finish initializing shared memory.
    __syncthreads();

    // Atomically add the values of the examination parameters of this thread's
    // query session to the shared examination parameters of this block.
    // Start every thread in this block at a different query session ranks
    // so prevent all threads from atomically writing to the same rank at the
    // same time.
    if (thread_index < partition_size) {
        int rank{0}, start_rank = block_index % MAX_SERP_LENGTH;
        for (int offset = 0; offset < MAX_SERP_LENGTH; offset++) {
            rank = (start_rank + offset) % MAX_SERP_LENGTH;

            // Atomically add the numerator and denominator values to shared memory.
            atomicAddArch(&block_examination_num[rank], this->tmp_examination_parameters[thread_index * MAX_SERP_LENGTH + rank].numerator_val());
            atomicAddArch(&block_examination_denom, 1.f / MAX_SERP_LENGTH);
        }
    }
    // Wait for all threads to finish writing to shared memory.
    __syncthreads();
    // Have only the first few threads of the block write the shared memory
    // results to global memory.
    if (block_index < MAX_SERP_LENGTH) {
        this->examination_parameters[block_index].atomic_add_to_values(block_examination_num[block_index], block_examination_denom);
    }
}

/**
 * @brief Update the global attractiveness parameters using the local
 * attractiveness parameters of a single thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of this thread.
 */
DEV void PBM_Dev::update_attractiveness_parameters(SERP& query_session, int& thread_index) {
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult sr = query_session[rank];
        this->attractiveness_parameters[sr.get_param_index()].atomic_add_to_values(
            this->tmp_attractiveness_parameters[thread_index * MAX_SERP_LENGTH + rank].numerator_val(),
            1.f);
    }
}
