/** CCM click model.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * ccm.cu:
 *  - Defines the functions specific to creating a CCM CM.
 */

#include "ccm.cuh"


//---------------------------------------------------------------------------//
// Host-side CCM click model functions.                                      //
//---------------------------------------------------------------------------//

HST CCM_Hst::CCM_Hst() = default;

/**
 * @brief Constructs a CCM click model object for the host.
 *
 * @param ccm
 * @returns CCM_Hst The CCM click model object.
 */
HST CCM_Hst::CCM_Hst(CCM_Hst const &ccm) {
}

/**
 * @brief Creates a new CCM click model object.
 *
 * @return CCM_Hst* The CCM click model object.
 */
HST CCM_Hst* CCM_Hst::clone() {
    return new CCM_Hst(*this);
}

/**
 * @brief Print a message.
 */
HST void CCM_Hst::say_hello() {
    std::cout << "Host-side CCM says hello!" << std::endl;
}

/**
 * @brief Get the amount of device memory allocated to this click model.
 *
 * @return size_t The used memory.
 */
HST size_t CCM_Hst::get_memory_usage(void) {
    return this->cm_memory_usage;
}

/**
 * @brief Get the expected amount of memory the click model will need to store
 * the current parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @return size_t The worst-case parameter memory footprint.
 */
HST size_t CCM_Hst::compute_memory_footprint(int n_queries, int n_qd) {
    std::pair<int, int> n_attractiveness = this->get_n_attr_params(n_queries, n_qd);
    std::pair<int, int> n_continuation = this->get_n_cont_params(n_queries, n_qd);

    return (n_attractiveness.first + n_attractiveness.second +
            n_continuation.first + n_continuation.second) * sizeof(Param);
}

/**
 * @brief Get the number of original and temporary attractiveness parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return std::pair<int,int> The number of original and temporary examination
 * parameters.
 */
HST std::pair<int,int> CCM_Hst::get_n_attr_params(int n_queries, int n_qd) {
    return std::make_pair(n_qd,                         // # original
                          n_queries * MAX_SERP_LENGTH); // # temporary
}

/**
 * @brief Get the number of original and temporary continuation parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return std::pair<int,int> The number of original and temporary continuation
 * parameters.
 */
HST std::pair<int, int> CCM_Hst::get_n_cont_params(int n_queries, int n_qd) {
    return std::make_pair(3,              // # original
                          n_queries * 3); // # temporary
}
/**
 * @brief Allocate device-side memory for the attractiveness parameters.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void CCM_Hst::init_attractiveness_parameters(const std::tuple<std::vector<SERP_HST>, std::vector<SERP_HST>, int>& partition, const size_t fmem) {
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
    this->tmp_attractiveness_parameters.resize(this->n_tmp_attr_dev, default_parameter);
    CUDA_CHECK(cudaMalloc(&this->tmp_attr_param_dptr, this->n_tmp_attr_dev * sizeof(Param)));
}

/**
 * @brief Allocate device-side memory for the continuation parameters, tau.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void CCM_Hst::init_tau_parameters(const std::tuple<std::vector<SERP_HST>, std::vector<SERP_HST>, int>& partition, const size_t fmem) {
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Compute the storage space required to store the parameters.
    std::pair<int, int> n_continuation = this->get_n_cont_params(std::get<0>(partition).size(), std::get<2>(partition));
    this->n_tau_dev = n_continuation.first;
    this->n_tmp_tau_dev = n_continuation.second;
    // Store the number of allocated bytes.
    this->cm_memory_usage += this->n_tau_dev * sizeof(Param) + this->n_tmp_tau_dev * sizeof(Param);
    // Check if the new parameters will fit in GPU memory using a 0.1% error margin.
    if (this->cm_memory_usage * 1.001 > fmem) {
        std::cout << "Error: Insufficient GPU memory!\n" <<
        "\tAllocating continuation parameters requires an additional " <<
        (this->cm_memory_usage - fmem) / 1e6 << " MB of GPU memory." << std::endl;
        mpi_abort(-1);
    }

    // Allocate memory for the continuation parameters on the device.
    this->tau_parameters.resize(this->n_tau_dev, default_parameter);
    CUDA_CHECK(cudaMalloc(&this->tau_param_dptr, this->n_tau_dev * sizeof(Param)));
    CUDA_CHECK(cudaMemcpy(this->tau_param_dptr, this->tau_parameters.data(),
                          this->n_tau_dev * sizeof(Param), cudaMemcpyHostToDevice));

    // Allocate memory for the temporary continuation parameters on the device.
    // These values are replaced at the start of each iteration, which means
    // they don't need to be initialized with a CUDA memory copy.
    this->tmp_tau_parameters.resize(this->n_tmp_tau_dev);
    CUDA_CHECK(cudaMalloc(&this->tmp_tau_param_dptr, this->n_tmp_tau_dev * sizeof(Param)));
}

/**
 * @brief Allocate device-side memory for the attractiveness and continuation
 * parameters of the click model.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void CCM_Hst::init_parameters(const std::tuple<std::vector<SERP_HST>, std::vector<SERP_HST>, int>& partition, const size_t fmem) {
    this->init_attractiveness_parameters(partition, fmem);
    this->init_tau_parameters(partition, fmem);
}

/**
 * @brief Get the references to the allocated device-side memory.
 *
 * @param param_refs An array containing the references to the device-side
 * parameters in memory.
 * @param param_sizes The size of each of the memory allocations on the device.
 */
HST void CCM_Hst::get_device_references(Param**& param_refs, int*& param_sizes) {
    int n_references = 4;

    // Create a temporary array to store the device references.
    Param* tmp_param_refs_array[n_references];
    tmp_param_refs_array[0] = this->attr_param_dptr;
    tmp_param_refs_array[1] = this->tmp_attr_param_dptr;
    tmp_param_refs_array[2] = this->tau_param_dptr;
    tmp_param_refs_array[3] = this->tmp_tau_param_dptr;

    // Allocate space for the device references.
    CUDA_CHECK(cudaMalloc(&param_refs, n_references * sizeof(Param*)));
    CUDA_CHECK(cudaMemcpy(param_refs, tmp_param_refs_array,
                          n_references * sizeof(Param*), cudaMemcpyHostToDevice));

    int tmp_param_sizes_array[n_references];
    tmp_param_sizes_array[0] = this->n_attr_dev;
    tmp_param_sizes_array[1] = this->n_tmp_attr_dev;
    tmp_param_sizes_array[2] = this->n_tau_dev;
    tmp_param_sizes_array[3] = this->n_tmp_tau_dev;

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
HST void CCM_Hst::update_parameters(int& gridSize, int& blockSize, SERP_DEV*& partition, int& dataset_size) {
    Kernel::update<<<gridSize, blockSize>>>(partition, dataset_size);
}

HST void CCM_Hst::update_parameters_on_host(const int& n_threads, const int* thread_start_idx, std::vector<SERP_HST>& partition){
    // Kernel::update<<<gridSize, blockSize>>>(partition, dataset_size);
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
HST void CCM_Hst::reset_parameters(void) {
    // Create a parameter initialized at 0.
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Create an array of the right proportions with the empty parameters.
    std::vector<Param> cleared_attractiveness_parameters(this->n_attr_dev, default_parameter);
    std::vector<Param> cleared_tau_parameters(this->n_tau_dev, default_parameter);

    // Copy the cleared array to the device.
    CUDA_CHECK(cudaMemcpy(this->attr_param_dptr, cleared_attractiveness_parameters.data(), this->n_attr_dev * sizeof(Param), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(this->tau_param_dptr, cleared_tau_parameters.data(), this->n_tau_dev * sizeof(Param), cudaMemcpyHostToDevice));
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
HST void CCM_Hst::transfer_parameters(int parameter_type, int transfer_direction) {
    // Public parameters.
    if (parameter_type == PUBLIC || parameter_type == ALL) {
        if (transfer_direction == D2H) { // Transfer from device to host.
            // Retrieve the continuation parameters from the device.
            CUDA_CHECK(cudaMemcpy(this->tau_parameters.data(), this->tau_param_dptr, this->n_tau_dev * sizeof(Param), cudaMemcpyDeviceToHost));
        }
        else if (transfer_direction == H2D) { // Transfer from host to device.
            // Send the continuation parameters to the device.
            CUDA_CHECK(cudaMemcpy(this->tau_param_dptr, this->tau_parameters.data(), this->n_tau_dev * sizeof(Param), cudaMemcpyHostToDevice));
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
HST void CCM_Hst::get_parameters(std::vector<std::vector<Param>>& destination, int parameter_type) {
    // Add the parameters to a generic vector which can represent  multiple
    // retrieved parameter types.
    if (parameter_type == PUBLIC) {
        destination.resize(1);
        destination[0] = this->tau_parameters;
    }
    else if (parameter_type == PRIVATE) {
        destination.resize(1);
        destination[0] = this->attractiveness_parameters;
    }
    else if (parameter_type == ALL) {
        destination.resize(2);
        destination[0] = this->attractiveness_parameters;
        destination[1] = this->tau_parameters;
    }
}

/**
 * @brief Compute the result of combining the CCM parameters from other nodes
 * or devices.
 *
 * @param parameters A multi-dimensional vector containing the parameters to be
 * combined. The vector is structured as follows: Node/Device ID -> Parameter
 * type -> Parameters.
 */
HST void CCM_Hst::sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters) {
    for (int r = 0; r < parameters[0][0].size(); r++) {
        for (int param_type = 0; param_type < parameters[0].size(); param_type++) {
            Param base = parameters[0][param_type][r];

            // Subtract the starting values of other partitions.
            parameters[0][param_type][r].set_values(base.numerator_val() - (parameters.size() - 1),
                                                    base.denominator_val() - 2 * (parameters.size() - 1));

            for (int device_id = 1; device_id < parameters.size(); device_id++) {
                Param ex = parameters[device_id][param_type][r];
                parameters[0][param_type][r].add_to_values(ex.numerator_val(),
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
HST void CCM_Hst::set_parameters(std::vector<std::vector<Param>>& source, int parameter_type) {
    // Set the parameters of this click model.
    if (parameter_type == PUBLIC) {
        this->tau_parameters = source[0];
    }
    else if (parameter_type == PRIVATE) {
        this->attractiveness_parameters = source[0];
    }
    else if (parameter_type == ALL) {
        this->attractiveness_parameters = source[0];
        this->tau_parameters = source[1];
    }
}

/**
 * @brief Compute the log-likelihood of the current CCM for the given query
 * session.
 *
 * @param query_session The query session for which the log-likelihood will be
 * computed.
 * @param log_click_probs The vector which will store the log-likelihood for
 * the document at each rank in the query session.
 */
HST void CCM_Hst::get_log_conditional_click_probs(SERP_HST& query_session, std::vector<float>& log_click_probs) {
    float atr, tau_1, tau_2, tau_3;
    float ex{1.f}, click_prob;

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult_HST sr = query_session[rank];

        atr = (float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM;
        if (sr.get_param_index() != -1)
            atr = this->attractiveness_parameters[sr.get_param_index()].value(); // Potential improvement: use thread_index as param_index. Store only chars about whether the document has been clicked or not.
        tau_1 = this->tau_parameters[0].value();
        tau_2 = this->tau_parameters[1].value();
        tau_3 = this->tau_parameters[2].value();

        if (sr.get_click() == 1) {
            click_prob = atr * ex;
            ex = tau_2 * (1 - atr) + tau_3 * atr;
        }
        else {
            click_prob = 1 - atr * ex;
            ex *= tau_1 * (1 - atr) / click_prob;
        }

        log_click_probs.push_back(std::log(click_prob));
    }
}

/**
 * @brief Compute the click probability of the current CCM for the given query
 * session.
 *
 * @param query_session The query session for which the click probability will
 * be computed.
 * @param full_click_probs The vector which will store the click probability
 * for the document at each rank in the query session.
 */
HST void CCM_Hst::get_full_click_probs(SERP_HST& query_session, std::vector<float> &full_click_probs) {
    float atr, tau_1, tau_2, tau_3;
    float ex{1.f}, atr_mul_ex;

    // Go through all ranks of the query session.
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        // Retrieve the search result at the current rank.
        SearchResult_HST sr = query_session[rank];

        atr = (float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM;
        if (sr.get_param_index() != -1)
            atr = this->attractiveness_parameters[sr.get_param_index()].value();
        tau_1 = this->tau_parameters[0].value();
        tau_2 = this->tau_parameters[1].value();
        tau_3 = this->tau_parameters[2].value();

        // Calculate the click probability.
        atr_mul_ex = atr * ex;

        // Calculate the full click probability.
        if (sr.get_click() == 1) {
            full_click_probs.push_back(atr_mul_ex);
        }
        else {
            full_click_probs.push_back(1 - atr_mul_ex);
        }

        ex *= (1 - atr) * tau_1 + atr * ((1 - atr) * tau_2 + atr * tau_3);
    }
}

/**
 * @brief Frees the memory allocated to the parameters of this click model on
 * the GPU device.
 */
HST void CCM_Hst::destroy_parameters(void) {
    // Free origin and temporary attractiveness containers.
    CUDA_CHECK(cudaFree(this->attr_param_dptr));
    CUDA_CHECK(cudaFree(this->tmp_attr_param_dptr));

    // Free origin and temporary continuation containers.
    CUDA_CHECK(cudaFree(this->tau_param_dptr));
    CUDA_CHECK(cudaFree(this->tmp_tau_param_dptr));

    // Free the device parameter references and sizes.
    CUDA_CHECK(cudaFree(this->param_refs));
    CUDA_CHECK(cudaFree(this->param_sizes));

    // Reset used device memory.
    this->cm_memory_usage = 0;
}


//---------------------------------------------------------------------------//
// Device-side CCM click model functions.                                    //
//---------------------------------------------------------------------------//

/**
 * @brief Prints a message.
 */
DEV void CCM_Dev::say_hello() {
    printf("Device-side CCM says hello!\n");
}

/**
 * @brief Creates a new CCM click model object.
 *
 * @return CCM_Dev* The CCM click model object.
 */
DEV CCM_Dev *CCM_Dev::clone() {
    return new CCM_Dev(*this);
}

DEV CCM_Dev::CCM_Dev() = default;

/**
 * @brief Constructs a CCM click model object for the device.
 *
 * @param ccm
 * @returns CCM_Dev The CCM click model object.
 */
DEV CCM_Dev::CCM_Dev(CCM_Dev const &ccm) {
}

/**
 * @brief Set the location of the memory allocated for the parameters of this
 * click model on the GPU device.
 *
 * @param parameter_ptr The pointers to the allocated memory.
 * @param parameter_sizes The size of the allocated memory.
 */
DEV void CCM_Dev::set_parameters(Param**& parameter_ptr, int* parameter_sizes) {
    // Set pointers to parameter arrays.
    this->attractiveness_parameters = parameter_ptr[0];
    this->tmp_attractiveness_parameters = parameter_ptr[1];
    this->tau_parameters = parameter_ptr[2];
    this->tmp_tau_parameters = parameter_ptr[3];

    // Set parameter array sizes.
    this->n_attractiveness_parameters = parameter_sizes[0];
    this->n_tmp_attractiveness_parameters = parameter_sizes[1];
    this->n_tau_parameters = parameter_sizes[2];
    this->n_tmp_tau_parameters = parameter_sizes[3];
}

/**
 * @brief Compute a single Expectation-Maximization iteration for the CCM click
 * model, for a single query session.
 *
 * @param query_session The query session which will be used to estimate the
 * CCM parameters.
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 */
DEV void CCM_Dev::process_session(SERP_DEV& query_session, int& thread_index, int& partition_size) {
    int last_click_rank = query_session.last_click_rank();
    float click_probs[MAX_SERP_LENGTH][MAX_SERP_LENGTH] = { 0.f };
    float exam_probs[MAX_SERP_LENGTH + 1];
    float exam[MAX_SERP_LENGTH + 1];
    float car[MAX_SERP_LENGTH + 1] = { 0.f };

    this->tmp_tau_parameters[thread_index].set_values(0.f, 0.f);
    this->tmp_tau_parameters[partition_size + thread_index].set_values(0.f, 0.f);
    this->tmp_tau_parameters[2 * partition_size + thread_index].set_values(0.f, 0.f);

    this->compute_exam_car(thread_index, query_session, exam, car);
    this->compute_ccm_attr(thread_index, query_session, last_click_rank, exam, car, partition_size);
    this->get_tail_clicks(thread_index, query_session, click_probs, exam_probs);
    this->compute_taus(thread_index, query_session, last_click_rank, click_probs, exam_probs, partition_size);
}

/**
 * @brief Compute the examination parameter for every rank of this query
 * session. The examination parameter can be re-computed every iteration using
 * the values from attractiveness, satisfaction, and continuation parameters
 * from the previous iteration.
 *
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param exam The examination parameters for every rank. The first rank is
 * always examined (1).
 * @param car
 */
DEV void CCM_Dev::compute_exam_car(int& thread_index, SERP_DEV& query_session, float (&exam)[MAX_SERP_LENGTH + 1], float (&car)[MAX_SERP_LENGTH + 1]) {
    // Set the default examination value for the first rank.
    exam[0] = 1.f;

    float attr_val, tau_1, tau_2, tau_3, ex_value, temp, car_val;
    float car_helper[MAX_SERP_LENGTH][2];

    for (int rank = 0; rank < MAX_SERP_LENGTH;) {
        SearchResult_DEV sr = query_session[rank];

        attr_val = this->attractiveness_parameters[sr.get_param_index()].value();
        tau_1 = this->tau_parameters[0].value();
        tau_2 = this->tau_parameters[1].value();
        tau_3 = this->tau_parameters[2].value();
        ex_value = exam[rank];

        temp = (1 - attr_val) * tau_1;

        // Calculate epsilon for the next rank.
        ex_value *= temp + attr_val * ((1 - attr_val) * tau_2 + attr_val * tau_3);

        car_helper[rank][0] = attr_val;
        car_helper[rank][1] = temp;

        // Set the examination value for the next rank.
        rank += 1;
        exam[rank] = ex_value;
    }

    for (int car_itr = MAX_SERP_LENGTH - 1; car_itr > -1; car_itr--) {
        car_val = car[car_itr + 1];

        car[car_itr] = car_helper[car_itr][0] + car_helper[car_itr][1] * car_val;
    }
}

/**
 * @brief Compute the attractiveness parameter for every rank of this query
 * session.
 *
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param exam The examination parameters for every rank. The first rank is
 * always examined (1).
 * @param car
 */
DEV void CCM_Dev::compute_ccm_attr(int& thread_index, SERP_DEV& query_session, int& last_click_rank, float (&exam)[MAX_SERP_LENGTH + 1], float (&car)[MAX_SERP_LENGTH + 1], int& partition_size) {
    float numerator_update, denominator_update;
    float attr_val, exam_val, car_val;

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult_DEV sr = query_session[rank];
        int click = sr.get_click();

        numerator_update = 0.f;
        denominator_update = 1.f;

        attr_val = this->attractiveness_parameters[sr.get_param_index()].value();
        exam_val = exam[rank];

        if (click == 1) {
            numerator_update += 1;
            denominator_update += 1;
        }
        else if (rank >= last_click_rank) {
            car_val = car[rank];
            numerator_update += ((1 - exam_val) * attr_val) / (1 - exam_val * car_val);
        }

        if (click == 1 && rank == last_click_rank) {
            car_val = car[rank + 1];
            numerator_update += attr_val / (1 - (this->tau_parameters[1].value() * (1 - attr_val) + this->tau_parameters[2].value() * attr_val) * car_val);
        }

        this->tmp_attractiveness_parameters[rank * partition_size + thread_index].set_values(numerator_update, denominator_update);
    }
}

/**
 * @brief Compute the click probabilities of a rank given the clicks on the
 * preceding ranks.
 *
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param click_probs The probabilty of a click occurring on a rank.
 * @param exam_probs The probability of a rank being examined.
 */
DEV void CCM_Dev::get_tail_clicks(int& thread_index, SERP_DEV& query_session, float (&click_probs)[MAX_SERP_LENGTH][MAX_SERP_LENGTH], float (&exam_probs)[MAX_SERP_LENGTH + 1]) {
    exam_probs[0] = 1.f;
    float tau_1, tau_2, tau_3;
    float exam_val, click_prob;

    for (int start_rank = 0; start_rank < MAX_SERP_LENGTH; start_rank++) {
        exam_val = 1.f;

        int ses_itr{0};
        for (int res_itr = start_rank; res_itr < MAX_SERP_LENGTH; res_itr++) {
            SearchResult_DEV tmp_sr = query_session[ses_itr];

            float attr_val = this->attractiveness_parameters[tmp_sr.get_param_index()].value();
            tau_1 = this->tau_parameters[0].value();
            tau_2 = this->tau_parameters[1].value();
            tau_3 = this->tau_parameters[2].value();

            if (query_session[res_itr].get_click() == 1) {
                click_prob = attr_val * exam_val;
                exam_val = tau_2 * (1 - attr_val) + tau_3 * attr_val;
            }
            else {
                click_prob = 1 - attr_val * exam_val;
                exam_val *= tau_1 * (1 - attr_val) / click_prob;
            }

            click_probs[start_rank][ses_itr] = click_prob;

            if (start_rank == 0) {
                exam_probs[ses_itr + 1] = exam_val;
            }

            ses_itr++;
        }
    }
}

/**
 * @brief Compute the continuation parameters tau 1, tau 2, and tau 3.
 *
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param click_probs The probabilty of a click occurring on a rank.
 * @param exam_probs The probability of a rank being examined.
 */
DEV void CCM_Dev::compute_taus(int& thread_index, SERP_DEV& query_session, int& last_click_rank, float (&click_probs)[MAX_SERP_LENGTH][MAX_SERP_LENGTH], float (&exam_probs)[MAX_SERP_LENGTH + 1], int& partition_size) {
    float factor_values[8] = { 0.f };

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++){
        SearchResult_DEV sr = query_session[rank];

        // Send the initialization values to the phi function.
        CCMFactor factor_func(click_probs, exam_probs, sr.get_click(),
                              last_click_rank, rank,
                              this->attractiveness_parameters[sr.get_param_index()].value(),
                              this->tau_parameters[0].value(),
                              this->tau_parameters[1].value(),
                              this->tau_parameters[2].value());

        float factor_result = 0.f;
        float factor_sum = 0.f;

        // Compute phi for all possible input values.
        for (int fct_itr{0}; fct_itr < 8; fct_itr++) {
            factor_result = factor_func.compute(this->factor_inputs[fct_itr][0],
                                                this->factor_inputs[fct_itr][1],
                                                this->factor_inputs[fct_itr][2]);
            factor_values[fct_itr] = factor_result;
            factor_sum += factor_result;
        }

        if (sr.get_click() == 0) {
            this->compute_tau_1(thread_index, factor_values, factor_sum, partition_size);
        }
        else {
            this->compute_tau_2(thread_index, factor_values, factor_sum, partition_size);
            this->compute_tau_3(thread_index, factor_values, factor_sum, partition_size);
        }
    }
}

DEV void CCM_Dev::compute_tau_1(int& thread_index, float (&factor_values)[8], float& factor_sum, int& partition_size) {
    double numerator_update{(factor_values[5] + factor_values[7]) / factor_sum};
    double denominator_update{numerator_update + ((factor_values[4] + factor_values[6]) / factor_sum)};
    this->tmp_tau_parameters[thread_index].add_to_values(numerator_update, denominator_update);
}

DEV void CCM_Dev::compute_tau_2(int& thread_index, float (&factor_values)[8], float& factor_sum, int& partition_size) {
    double numerator_update{factor_values[5] / factor_sum};
    double denominator_update{numerator_update + ((factor_values[4]) / factor_sum)};
    this->tmp_tau_parameters[partition_size + thread_index].add_to_values(numerator_update, denominator_update);
}

DEV void CCM_Dev::compute_tau_3(int& thread_index, float (&factor_values)[8], float& factor_sum, int& partition_size) {
    double numerator_update{factor_values[7] / factor_sum};
    double denominator_update{numerator_update + ((factor_values[6]) / factor_sum)};
    this->tmp_tau_parameters[2 * partition_size + thread_index].add_to_values(numerator_update, denominator_update);
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
DEV void CCM_Dev::update_parameters(SERP_DEV& query_session, int& thread_index, int& block_index, int& partition_size) {
    this->update_tau_parameters(query_session, thread_index, block_index, partition_size);

    if (thread_index < partition_size) {
        this->update_attractiveness_parameters(query_session, thread_index, partition_size);
    }
}

/**
 * @brief Update the global continuation parameters using the local continuation
 * parameters of a single thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of this thread.
 * @param block_index The index of the block in which this thread exists.
 * @param partition_size The size of the dataset.
 */
DEV void CCM_Dev::update_tau_parameters(SERP_DEV& query_session, int& thread_index, int& block_index, int& partition_size) {
    SHR float numerator[BLOCK_SIZE];
    SHR float denominator[BLOCK_SIZE];

    for (int tau_num = 0; tau_num < 3; tau_num++) {
        // Initialize shared memory for this block's parameters.
        if (thread_index < partition_size) {
            Param tmp_param = this->tmp_tau_parameters[tau_num * partition_size + thread_index];
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
            this->tau_parameters[tau_num].atomic_add_to_values(numerator[0],
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
DEV void CCM_Dev::update_attractiveness_parameters(SERP_DEV& query_session, int& thread_index, int& partition_size) {
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult_DEV sr = query_session[rank];
        Param atr_update = this->tmp_attractiveness_parameters[rank * partition_size + thread_index];
        this->attractiveness_parameters[sr.get_param_index()].atomic_add_to_values(atr_update.numerator_val(),
                                                                                   atr_update.denominator_val());
    }
}
