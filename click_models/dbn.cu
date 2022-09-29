/** DBN click model.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * dbn.cu:
 *  - Defines the functions specific to creating a DBN CM.
 */

#include "dbn.cuh"


//---------------------------------------------------------------------------//
// Host-side DBN click model functions.                                      //
//---------------------------------------------------------------------------//

HST DBN_Host::DBN_Host() = default;

/**
 * @brief Constructs a DBN click model object for the host.
 *
 * @param dbn
 * @returns DBN_Host The DBN click model object.
 */
HST DBN_Host::DBN_Host(DBN_Host const &dbn) {
}

/**
 * @brief Creates a new DBN click model object.
 *
 * @return DBN_Host* The DBN click model object.
 */
HST DBN_Host* DBN_Host::clone() {
    return new DBN_Host(*this);
}

/**
 * @brief Print a message.
 */
HST void DBN_Host::say_hello() {
    std::cout << "Host-side DBN says hello!" << std::endl;
}

/**
 * @brief Get the amount of device memory allocated to this click model.
 *
 * @return size_t The used memory.
 */
HST size_t DBN_Host::get_memory_usage(void) {
    return this->cm_memory_usage;
}

/**
 * @brief Allocate device-side memory for the attractiveness parameters.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void DBN_Host::init_attractiveness_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, int n_devices) {
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
 * @brief Allocate device-side memory for the satisfaction parameters.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void DBN_Host::init_satisfaction_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, int n_devices) {
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Allocate memory for the satisfaction parameters on the device.
    this->n_satisfaction_dev = std::get<2>(partition);
    this->satisfaction_parameters.resize(this->n_satisfaction_dev, default_parameter);
    CUDA_CHECK(cudaMalloc(&this->satisfaction_param_dptr, this->n_satisfaction_dev * sizeof(Param)));
    CUDA_CHECK(cudaMemcpy(this->satisfaction_param_dptr, this->satisfaction_parameters.data(),
                          this->n_satisfaction_dev * sizeof(Param), cudaMemcpyHostToDevice));

    // Allocate memory for the temporary satisfaction parameters on the device.
    // These values are replaced at the start of each iteration, which means
    // they don't need to be initialized with a CUDA memory copy.
    this->n_tmp_satisfaction_dev = std::get<0>(partition).size() * MAX_SERP_LENGTH;
    this->tmp_satisfaction_parameters.resize(this->n_tmp_satisfaction_dev);
    CUDA_CHECK(cudaMalloc(&this->tmp_satisfaction_param_dptr, this->n_tmp_satisfaction_dev * sizeof(Param)));

    // Store the number of allocated bytes.
    this->cm_memory_usage += this->n_satisfaction_dev * sizeof(Param) + this->n_tmp_satisfaction_dev * sizeof(Param);
}

/**
 * @brief Allocate device-side memory for the continuation parameters, gamma.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void DBN_Host::init_gamma_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, int n_devices) {
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Allocate memory for the continuation parameters on the device.
    this->n_gamma_dev = 1;
    this->gamma_parameters.resize(this->n_gamma_dev, default_parameter);
    CUDA_CHECK(cudaMalloc(&this->gamma_param_dptr, this->n_gamma_dev * sizeof(Param)));
    CUDA_CHECK(cudaMemcpy(this->gamma_param_dptr, this->gamma_parameters.data(),
                          this->n_gamma_dev * sizeof(Param), cudaMemcpyHostToDevice));

    // Allocate memory for the temporary continuation parameters on the device.
    // These values are replaced at the start of each iteration, which means
    // they don't need to be initialized with a CUDA memory copy.
    this->n_tmp_gamma_dev = std::get<0>(partition).size() * this->n_gamma_dev;
    this->tmp_gamma_parameters.resize(this->n_tmp_gamma_dev);
    CUDA_CHECK(cudaMalloc(&this->tmp_gamma_param_dptr, this->n_tmp_gamma_dev * sizeof(Param)));

    // Store the number of allocated bytes.
    this->cm_memory_usage += this->n_gamma_dev * sizeof(Param) + this->n_tmp_gamma_dev * sizeof(Param);
}

/**
 * @brief Allocate device-side memory for the attractiveness, satisfaction and
 * continuation parameters of the click model.
 *
 * @param partition The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 */
HST void DBN_Host::init_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, int n_devices) {
    this->init_attractiveness_parameters(partition, n_devices);
    this->init_satisfaction_parameters(partition, n_devices);
    this->init_gamma_parameters(partition, n_devices);
}

/**
 * @brief Get the references to the allocated device-side memory.
 *
 * @param param_refs An array containing the references to the device-side
 * parameters in memory.
 * @param param_sizes The size of each of the memory allocations on the device.
 */
HST void DBN_Host::get_device_references(Param**& param_refs, int*& param_sizes) {
    int n_references = 6;

    // Create a temporary array to store the device references.
    Param* tmp_param_refs_array[n_references];
    tmp_param_refs_array[0] = this->attr_param_dptr;
    tmp_param_refs_array[1] = this->tmp_attr_param_dptr;
    tmp_param_refs_array[2] = this->satisfaction_param_dptr;
    tmp_param_refs_array[3] = this->tmp_satisfaction_param_dptr;
    tmp_param_refs_array[4] = this->gamma_param_dptr;
    tmp_param_refs_array[5] = this->tmp_gamma_param_dptr;

    // Allocate space for the device references.
    CUDA_CHECK(cudaMalloc(&param_refs, n_references * sizeof(Param*)));
    CUDA_CHECK(cudaMemcpy(param_refs, tmp_param_refs_array,
                          n_references * sizeof(Param*), cudaMemcpyHostToDevice));

    int tmp_param_sizes_array[n_references];
    tmp_param_sizes_array[0] = this->n_attr_dev;
    tmp_param_sizes_array[1] = this->n_tmp_attr_dev;
    tmp_param_sizes_array[2] = this->n_satisfaction_dev;
    tmp_param_sizes_array[3] = this->n_tmp_satisfaction_dev;
    tmp_param_sizes_array[4] = this->n_gamma_dev;
    tmp_param_sizes_array[5] = this->n_tmp_gamma_dev;

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
HST void DBN_Host::update_parameters(int& gridSize, int& blockSize, SERP*& partition, int& dataset_size) {
    Kernel::update<<<gridSize, blockSize>>>(partition, dataset_size, 0);
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
HST void DBN_Host::reset_parameters(void) {
    // Create a parameter initialized at 0.
    Param default_parameter;
    default_parameter.set_values(PARAM_DEF_NUM, PARAM_DEF_DENOM);

    // Create an array of the right proportions with the empty parameters.
    std::vector<Param> cleared_attractiveness_parameters(this->n_attr_dev, default_parameter);
    std::vector<Param> cleared_satisfaction_parameters(this->n_satisfaction_dev, default_parameter);
    std::vector<Param> cleared_gamma_parameters(this->n_gamma_dev, default_parameter);

    // Copy the cleared array to the device.
    CUDA_CHECK(cudaMemcpy(this->attr_param_dptr, cleared_attractiveness_parameters.data(), this->n_attr_dev * sizeof(Param), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(this->satisfaction_param_dptr, cleared_satisfaction_parameters.data(), this->n_satisfaction_dev * sizeof(Param), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(this->gamma_param_dptr, cleared_gamma_parameters.data(), this->n_gamma_dev * sizeof(Param), cudaMemcpyHostToDevice));
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
HST void DBN_Host::transfer_parameters(int parameter_type, int transfer_direction) {
    // Public parameters.
    if (parameter_type == PUBLIC || parameter_type == ALL) {
        if (transfer_direction == D2H) { // Transfer from device to host.
            // Retrieve the continuation parameters from the device.
            CUDA_CHECK(cudaMemcpy(this->gamma_parameters.data(), this->gamma_param_dptr, this->n_gamma_dev * sizeof(Param), cudaMemcpyDeviceToHost));
        }
        else if (transfer_direction == H2D) { // Transfer from host to device.
            // Send the continuation parameters to the device.
            CUDA_CHECK(cudaMemcpy(this->gamma_param_dptr, this->gamma_parameters.data(), this->n_gamma_dev * sizeof(Param), cudaMemcpyHostToDevice));
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

        if (transfer_direction == D2H) { // Transfer from device to host.
            // Retrieve the satisfaction parameters from the device.
            CUDA_CHECK(cudaMemcpy(this->satisfaction_parameters.data(), this->satisfaction_param_dptr, this->n_satisfaction_dev * sizeof(Param), cudaMemcpyDeviceToHost));
        }
        else if (transfer_direction == H2D) { // Transfer from host to device.
            // Send the satisfaction parameters to the device.
            CUDA_CHECK(cudaMemcpy(this->satisfaction_param_dptr, this->satisfaction_parameters.data(), this->n_satisfaction_dev * sizeof(Param), cudaMemcpyHostToDevice));
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
HST void DBN_Host::get_parameters(std::vector<std::vector<Param>>& destination, int parameter_type) {
    // Add the parameters to a generic vector which can represent  multiple
    // retrieved parameter types.
    if (parameter_type == PUBLIC) {
        destination.resize(1);
        destination[0] = this->gamma_parameters;
    }
    else if (parameter_type == PRIVATE) {
        destination.resize(2);
        destination[0] = this->attractiveness_parameters;
        destination[1] = this->satisfaction_parameters;
    }
    else if (parameter_type == ALL) {
        destination.resize(3);
        destination[0] = this->attractiveness_parameters;
        destination[1] = this->satisfaction_parameters;
        destination[2] = this->gamma_parameters;
    }
}

/**
 * @brief Compute the result of combining the DBN parameters from other nodes
 * or devices.
 *
 * @param parameters A multi-dimensional vector containing the parameters to be
 * combined. The vector is structured as follows: Node/Device ID -> Parameter
 * type -> Parameters.
 */
HST void DBN_Host::sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters) {
    for (int r = 0; r < parameters[0][0].size(); r++) {
        for (int param_type = 0; param_type < parameters[0].size(); param_type++) {
            // Use the first sub-array to combine the results in.
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
HST void DBN_Host::set_parameters(std::vector<std::vector<Param>>& source, int parameter_type) {
    // Set the parameters of this click model.
    if (parameter_type == PUBLIC) {
        this->gamma_parameters = source[0];
    }
    else if (parameter_type == PRIVATE) {
        this->attractiveness_parameters = source[0];
        this->satisfaction_parameters = source[1];
    }
    else if (parameter_type == ALL) {
        this->attractiveness_parameters = source[0];
        this->satisfaction_parameters = source[1];
        this->gamma_parameters = source[2];
    }
}

/**
 * @brief Compute the log-likelihood of the current DBN for the given query
 * session.
 *
 * @param query_session The query session for which the log-likelihood will be
 * computed.
 * @param log_click_probs The vector which will store the log-likelihood for
 * the document at each rank in the query session.
 */
HST void DBN_Host::get_log_conditional_click_probs(SERP& query_session, std::vector<float>& log_click_probs) {
    float ex{1.f}, click_prob;

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float attr_val{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        float sat_val{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1) {
            attr_val = this->attractiveness_parameters[sr.get_param_index()].value();
            sat_val = this->satisfaction_parameters[sr.get_param_index()].value();
        }
        float gamma_val{this->gamma_parameters[0].value()};

        if (sr.get_click() == 1) {
            click_prob = attr_val * ex;
            ex = gamma_val * ( 1- sat_val);
        } else{
            click_prob = 1 - attr_val * ex;
            ex *= gamma_val * ( 1 - attr_val) / click_prob;
        }

        log_click_probs.push_back(std::log(click_prob));
    }
}

/**
 * @brief Compute the click probability of the current DBN for the given query
 * session.
 *
 * @param query_session The query session for which the click probability will
 * be computed.
 * @param full_click_probs The vector which will store the click probability
 * for the document at each rank in the query session.
 */
HST void DBN_Host::get_full_click_probs(SERP& query_session, std::vector<float> &full_click_probs) {
    float ex{1.f}, atr_mul_ex;

    // Go through all ranks of the query session.
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        // Retrieve the search result at the current rank.
        SearchResult sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float atr{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        float sat{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1) {
            atr = this->attractiveness_parameters[sr.get_param_index()].value();
            sat = this->satisfaction_parameters[sr.get_param_index()].value();
        }
        float gamma{this->gamma_parameters[0].value()};

        // Calculate the click probability.
        atr_mul_ex = atr * ex;

        // Calculate the full click probability.
        if (sr.get_click() == 1) {
            full_click_probs.push_back(atr_mul_ex);
        }
        else {
            full_click_probs.push_back(1 - atr_mul_ex);
        }

        ex *= gamma * (1 - atr) + gamma * atr * (1 - sat);
    }
}

/**
 * @brief Frees the memory allocated to the parameters of this click model on
 * the GPU device.
 */
HST void DBN_Host::destroy_parameters(void) {
    // Free origin and temporary attractiveness containers.
    CUDA_CHECK(cudaFree(this->attr_param_dptr));
    CUDA_CHECK(cudaFree(this->tmp_attr_param_dptr));

    // Free origin and temporary satisfaction containers.
    CUDA_CHECK(cudaFree(this->satisfaction_param_dptr));
    CUDA_CHECK(cudaFree(this->tmp_satisfaction_param_dptr));

    // Free origin and temporary continuation containers.
    CUDA_CHECK(cudaFree(this->gamma_param_dptr));
    CUDA_CHECK(cudaFree(this->tmp_gamma_param_dptr));

    // Free the device parameter references and sizes.
    CUDA_CHECK(cudaFree(this->param_refs));
    CUDA_CHECK(cudaFree(this->param_sizes));

    // Reset used device memory.
    this->cm_memory_usage = 0;
}


//---------------------------------------------------------------------------//
// Device-side DBN click model functions.                                    //
//---------------------------------------------------------------------------//

/**
 * @brief Prints a message.
 */
DEV void DBN_Dev::say_hello() {
    printf("Device-side DBN says hello!\n");
}

/**
 * @brief Creates a new DBN click model object.
 *
 * @return DBN_Dev* The DBN click model object.
 */
DEV DBN_Dev *DBN_Dev::clone() {
    return new DBN_Dev(*this);
}

DEV DBN_Dev::DBN_Dev() = default;

/**
 * @brief Constructs a DBN click model object for the device.
 *
 * @param dbn
 * @returns DBN_Dev The DBN click model object.
 */
DEV DBN_Dev::DBN_Dev(DBN_Dev const &dbn) {
}

/**
 * @brief Set the location of the memory allocated for the parameters of this
 * click model on the GPU device.
 *
 * @param parameter_ptr The pointers to the allocated memory.
 * @param parameter_sizes The size of the allocated memory.
 */
DEV void DBN_Dev::set_parameters(Param**& parameter_ptr, int* parameter_sizes) {
    // Set pointers to parameter arrays.
    this->attractiveness_parameters = parameter_ptr[0];
    this->tmp_attractiveness_parameters = parameter_ptr[1];
    this->satisfaction_parameters = parameter_ptr[2];
    this->tmp_satisfaction_parameters = parameter_ptr[3];
    this->gamma_parameters = parameter_ptr[4];
    this->tmp_gamma_parameters = parameter_ptr[5];

    // Set parameter array sizes.
    this->n_attractiveness_parameters = parameter_sizes[0];
    this->n_tmp_attractiveness_parameters = parameter_sizes[1];
    this->n_satisfaction_parameters = parameter_sizes[2];
    this->n_tmp_satisfaction_parameters = parameter_sizes[3];
    this->n_gamma_parameters = parameter_sizes[4];
    this->n_tmp_gamma_parameters = parameter_sizes[5];
}

/**
 * @brief Compute a single Expectation-Maximization iteration for the DBN click
 * model, for a single query session.
 *
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 */
DEV void DBN_Dev::process_session(SERP& query_session, int& thread_index) {
    int last_click_rank = query_session.last_click_rank();
    float click_probs[MAX_SERP_LENGTH][MAX_SERP_LENGTH] = { 0.f };
    float exam_probs[MAX_SERP_LENGTH + 1];
    float exam[MAX_SERP_LENGTH + 1];
    float car[MAX_SERP_LENGTH + 1] = { 0.f };

    this->tmp_gamma_parameters[thread_index].set_values(0.f, 0.f);

    this->compute_exam_car(thread_index, query_session, exam, car);
    this->compute_dbn_attr(thread_index, query_session, last_click_rank, exam, car);
    this->compute_dbn_sat(thread_index, query_session, last_click_rank, car);
    this->get_tail_clicks(thread_index, query_session, click_probs, exam_probs);
    this->compute_gamma(thread_index, query_session, last_click_rank, click_probs, exam_probs);
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
DEV void DBN_Dev::compute_exam_car(int& thread_index, SERP& query_session, float (&exam)[MAX_SERP_LENGTH + 1], float (&car)[MAX_SERP_LENGTH + 1]) {
    // Set the default examination value for the first rank.
    exam[0] = 1.f;

    float attr_val, sat_value, gamma_value, ex_value, temp, car_val;
    float car_helper[MAX_SERP_LENGTH][2];

    for (int rank = 0; rank < MAX_SERP_LENGTH;) {
        SearchResult sr = query_session[rank];

        attr_val = this->attractiveness_parameters[sr.get_param_index()].value();
        sat_value = this->satisfaction_parameters[sr.get_param_index()].value();
        gamma_value = this->gamma_parameters[0].value();
        ex_value = exam[rank];

        temp = gamma_value * (1 - attr_val);
        ex_value *= temp + gamma_value * attr_val * (1 - sat_value);

        car_helper[rank][0] = attr_val;
        car_helper[rank][1] = temp;

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
DEV void DBN_Dev::compute_dbn_attr(int& thread_index, SERP& query_session, int& last_click_rank, float (&exam)[MAX_SERP_LENGTH + 1], float (&car)[MAX_SERP_LENGTH + 1]) {
    float numerator_update, denominator_update;
    float exam_val, attr_val,  car_val;

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult sr = query_session[rank];

        numerator_update = 0.f;
        denominator_update = 1.f;

        if (sr.get_click() == 1) {
            numerator_update += 1.f;
        }
        else if (rank >= last_click_rank) {
            attr_val = this->attractiveness_parameters[sr.get_param_index()].value();
            exam_val = exam[rank];
            car_val = car[rank];

            numerator_update += (attr_val * (1 - exam_val)) / (1 - exam_val * car_val);
        }

        this->tmp_attractiveness_parameters[thread_index * MAX_SERP_LENGTH + rank].set_values(numerator_update, denominator_update);
    }
}

/**
 * @brief Compute the satisfaction parameter for every rank of this query
 * session.
 *
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param car
 */
DEV void DBN_Dev::compute_dbn_sat(int& thread_index, SERP& query_session, int& last_click_rank, float (&car)[MAX_SERP_LENGTH + 1]) {
    float numerator_update, denominator_update;
    float gamma_val, sat_val, car_val;

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult sr = query_session[rank];

        if (sr.get_click() == 1) {
            numerator_update = 0.f;
            denominator_update = 1.f;

            if (rank == last_click_rank){
                sat_val = this->satisfaction_parameters[sr.get_param_index()].value();
                gamma_val = this->gamma_parameters[0].value();

                if (rank < MAX_SERP_LENGTH - 1) {
                    car_val = car[rank + 1];
                } else{
                    car_val = 0.f;
                }

                numerator_update += sat_val / (1 - (1 - sat_val) * gamma_val * car_val);
            }

            this->tmp_satisfaction_parameters[thread_index * MAX_SERP_LENGTH + rank].set_values(numerator_update, denominator_update);
        }
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
DEV void DBN_Dev::get_tail_clicks(int& thread_index, SERP& query_session, float (&click_probs)[MAX_SERP_LENGTH][MAX_SERP_LENGTH], float (&exam_probs)[MAX_SERP_LENGTH + 1]) {
    exam_probs[0] = 1.f;
    float exam_val, gamma_val, click_prob;

    for (int start_rank = 0; start_rank < MAX_SERP_LENGTH; start_rank++) {
        exam_val = 1.f;

        int ses_itr{0};
        for (int res_itr = start_rank; res_itr < MAX_SERP_LENGTH; res_itr++, ses_itr++) {
            SearchResult tmp_sr = query_session[ses_itr];

            float attr_val = this->attractiveness_parameters[tmp_sr.get_param_index()].value();
            float sat_val = this->satisfaction_parameters[tmp_sr.get_param_index()].value();
            gamma_val = this->gamma_parameters[0].value();

            if (query_session[res_itr].get_click() == 1){
                click_prob = attr_val * exam_val;
                exam_val = gamma_val * (1 - sat_val);
            }
            else{
                click_prob = 1 - attr_val * exam_val;
                exam_val *= gamma_val * (1 - attr_val) / click_prob;
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
 * @brief Compute the continuation parameter gamma.
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
DEV void DBN_Dev::compute_gamma(int& thread_index, SERP& query_session, int& last_click_rank, float (&click_probs)[MAX_SERP_LENGTH][MAX_SERP_LENGTH], float (&exam_probs)[MAX_SERP_LENGTH + 1]) {
    float factor_values[8] = { 0.f };

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++){
        SearchResult sr = query_session[rank];

        // Send the initialization values to the phi function.
        DBNFactor factor_func(click_probs, exam_probs, sr.get_click(),
                              last_click_rank, rank,
                              this->attractiveness_parameters[sr.get_param_index()].value(),
                              this->satisfaction_parameters[sr.get_param_index()].value(),
                              this->gamma_parameters[0].value());

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

        float numerator_update = factor_values[5] / factor_sum;
        float denominator_update = (factor_values[4] + factor_values[5]) / factor_sum;

        this->tmp_gamma_parameters[thread_index].add_to_values(numerator_update, denominator_update);
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
DEV void DBN_Dev::update_parameters(SERP& query_session, int& thread_index, int& block_index, int& parameter_type, int& partition_size) {
    this->update_gamma_parameters(query_session, thread_index, block_index, partition_size);

    if (thread_index < partition_size) {
        this->update_attractiveness_parameters(query_session, thread_index);
        this->update_satisfaction_parameters(query_session, thread_index);
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
DEV void DBN_Dev::update_gamma_parameters(SERP& query_session, int& thread_index, int& block_index, int& partition_size) {
    // Initialize shared memory for this block's continuation parameters at 0.
    SHR float block_gamma_num;
    SHR float block_gamma_denom;
    block_gamma_num = 0.f;
    block_gamma_denom = 0.f;
    // Wait for all threads to finish initializing shared memory.
    __syncthreads();

    // Atomically add the values of the continuation parameters of this thread's
    // query session to the shared continuation parameters of this block.
    // Start every thread in this block at a different query session ranks
    // so prevent all threads from atomically writing to the same rank at the
    // same time.
    if (thread_index < partition_size) {
        // Atomically add the numerator and denominator values to shared memory.
        atomicAddArch(&block_gamma_num, this->tmp_gamma_parameters[thread_index].numerator_val());
        atomicAddArch(&block_gamma_denom, this->tmp_gamma_parameters[thread_index].denominator_val());
    }
    // Wait for all threads to finish writing to shared memory.
    __syncthreads();

    // Have only the first thread of the block write the shared memory
    // results to global memory.
    if (block_index == 0) {
        this->gamma_parameters[block_index].atomic_add_to_values(block_gamma_num, block_gamma_denom);
    }
}

/**
 * @brief Update the global attractiveness parameters using the local
 * attractiveness parameters of a single thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of this thread.
 */
DEV void DBN_Dev::update_attractiveness_parameters(SERP& query_session, int& thread_index) {
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult sr = query_session[rank];
        this->attractiveness_parameters[sr.get_param_index()].atomic_add_to_values(
            this->tmp_attractiveness_parameters[thread_index * MAX_SERP_LENGTH + rank].numerator_val(),
            this->tmp_attractiveness_parameters[thread_index * MAX_SERP_LENGTH + rank].denominator_val());
    }
}

/**
 * @brief Update the global satisfaction parameters using the local satisfaction
 * parameters of a single thread.
 *
 * @param query_session The query session of this thread.
 * @param thread_index The index of this thread.
 * @param block_index The index of the block in which this thread exists.
 * @param partition_size The size of the dataset.
 */
DEV void DBN_Dev::update_satisfaction_parameters(SERP& query_session, int& thread_index) {
    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        SearchResult sr = query_session[rank];
        this->satisfaction_parameters[sr.get_param_index()].atomic_add_to_values(
            this->tmp_satisfaction_parameters[thread_index * MAX_SERP_LENGTH + rank].numerator_val(),
            this->tmp_satisfaction_parameters[thread_index * MAX_SERP_LENGTH + rank].denominator_val());
    }
}