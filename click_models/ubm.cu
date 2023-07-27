/** UBM click model.
 *
 * ubm.cu:
 *  - Defines the functions specific to creating a UBM CM.
 */

#include "ubm.cuh"


//---------------------------------------------------------------------------//
// Host-side UBM click model functions.                                      //
//---------------------------------------------------------------------------//

HST UBM_Hst::UBM_Hst() = default;

/**
 * @brief Constructs a UBM click model object for the host.
 *
 * @param ubm The base click model object to copy.
 * @return The UBM click model object.
 */
HST UBM_Hst::UBM_Hst(UBM_Hst const &ubm) {
}

/**
 * @brief Creates a new UBM click model object.
 *
 * @return The UBM click model object.
 */
HST UBM_Hst* UBM_Hst::clone() {
    return new UBM_Hst(*this);
}

/**
 * @brief Print a message.
 */
HST void UBM_Hst::say_hello() {
    std::cout << "Host-side UBM says hello!" << std::endl;
}

/**
 * @brief Get the amount of device memory allocated to this click model.
 *
 * @return The used memory.
 */
HST size_t UBM_Hst::get_memory_usage(void) {
    return this->cm_memory_usage;
}

/**
 * @brief Get the expected amount of memory the click model will need to store
 * the current parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return The worst-case parameter memory footprint.
 */
HST size_t UBM_Hst::compute_memory_footprint(int n_queries, int n_qd) {
    std::pair<int, int> n_attractiveness = this->get_n_atr_params(n_queries, n_qd);
    std::pair<int, int> n_examination = this->get_n_exm_params(n_queries, n_qd);

    return (n_attractiveness.first + n_attractiveness.second +
            n_examination.first + n_examination.second) * sizeof(Param);
}

/**
 * @brief Get the number of original and temporary attractiveness parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return The number of original and temporary attractiveness
 * parameters.
 */
HST std::pair<int,int> UBM_Hst::get_n_atr_params(int n_queries, int n_qd) {
    return std::make_pair(n_qd,                  // #original.
                          n_queries * MAX_SERP); // #temporary.
}

/**
 * @brief Get the number of original and temporary examination parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return The number of original and temporary examination
 * parameters.
 */
HST std::pair<int, int> UBM_Hst::get_n_exm_params(int n_queries, int n_qd) {
    return std::make_pair((MAX_SERP - 1) * MAX_SERP / 2 + MAX_SERP,                // #original.
                          n_queries * ((MAX_SERP - 1) * MAX_SERP / 2 + MAX_SERP)); // #temporary.
}

/**
 * @brief Allocate device-side memory for the attractiveness and examination
 * parameters of the click model.
 *
 * @param dataset The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 * @param fmem The amount of free memory on the device.
 * @param device The device to allocate memory on.
 */
HST void UBM_Hst::init_parameters(const Partition& dataset, const size_t fmem, const bool device) {
    std::pair<int, int> n_attractiveness = this->get_n_atr_params(std::get<0>(dataset).size(), std::get<2>(dataset));
    init_parameters_hst(this->atr_parameters, this->atr_tmp_parameters, this->atr_dptr, this->atr_tmp_dptr, n_attractiveness, this->n_atr_params, this->n_atr_tmp_params, this->cm_memory_usage, dataset, fmem, device);
    std::pair<int, int> n_examination = this->get_n_exm_params(std::get<0>(dataset).size(), std::get<2>(dataset));
    init_parameters_hst(this->exm_parameters, this->exm_tmp_parameters, this->exm_dptr, this->exm_tmp_dptr, n_examination, this->n_exm_params, this->n_exm_tmp_params, this->cm_memory_usage, dataset, fmem, device);
}

/**
 * @brief Get the name of the parameters of this click model.
 *
 * @return The public and private parameter names.
 */
HST void UBM_Hst::get_parameter_information(
        std::pair<std::vector<std::string>, std::vector<std::string>> &headers,
        std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> &parameters) {
    // Set parameter headers.
    std::vector<std::string> public_name = {"examination"};
    std::vector<std::string> private_name = {"attractiveness"};
    headers = std::make_pair(public_name, private_name);

    // Set parameter values.
    std::vector<std::vector<Param> *> public_parameters = {&this->exm_parameters};
    std::vector<std::vector<Param> *> private_parameters = {&this->atr_parameters};
    parameters = std::make_pair(public_parameters, private_parameters);
}

/**
 * @brief Get the references to the allocated device-side memory.
 *
 * @param param_refs An array containing the references to the device-side
 * parameters in memory.
 * @param param_sizes The size of each of the memory allocations on the device.
 */
HST void UBM_Hst::get_device_references(Param**& param_refs, int*& param_sizes) {
    int n_references = 4;

    // Create a temporary array to store the device references.
    Param* tmp_param_refs_array[n_references];
    tmp_param_refs_array[0] = this->atr_dptr;
    tmp_param_refs_array[1] = this->atr_tmp_dptr;
    tmp_param_refs_array[2] = this->exm_dptr;
    tmp_param_refs_array[3] = this->exm_tmp_dptr;

    // Allocate space for the device references.
    CUDA_CHECK(cudaMalloc(&param_refs, n_references * sizeof(Param*)));
    CUDA_CHECK(cudaMemcpy(param_refs, tmp_param_refs_array,
                          n_references * sizeof(Param*), cudaMemcpyHostToDevice));

    int tmp_param_sizes_array[n_references];
    tmp_param_sizes_array[0] = this->n_atr_params;
    tmp_param_sizes_array[1] = this->n_atr_tmp_params;
    tmp_param_sizes_array[2] = this->n_exm_params;
    tmp_param_sizes_array[3] = this->n_exm_tmp_params;

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
 * @brief Update the global parameter values using the temporary parameters.
 *
 * @param dataset The training set.
 * @param thread_start_idx Dataset starting indices of each thread.
 */
HST void UBM_Hst::update_parameters(TrainSet& dataset, const std::vector<int>& thread_start_idx) {
    update_unique_parameters_hst(this->atr_tmp_parameters, this->atr_parameters, dataset, thread_start_idx);
    update_shared_parameters_hst(this->exm_tmp_parameters, this->exm_parameters, dataset, thread_start_idx);
}

/**
 * @brief Compute a single Expectation-Maximization iteration for the UBM click
 * model for each query session.
 *
 * @param dataset The training set.
 * @param thread_start_idx Dataset starting indices of each thread.
 */
HST void UBM_Hst::process_session(const TrainSet& dataset, const std::vector<int>& thread_start_idx) {
    // Iterate over the queries in the dataset in each thread.
    auto process_session_thread = [this](const TrainSet& dataset, const int thread_id, int start_idx, int stop_idx) {
        int dataset_size = dataset.size();

        for (int query_index = start_idx; query_index < stop_idx; query_index++) {
            SERP_Hst query_session = dataset[query_index];
            int prev_click_rank[MAX_SERP] = { 0 };
            query_session.prev_clicked_rank(prev_click_rank);

            #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
                SearchResult_Hst sr = query_session[rank];

                // Get the attractiveness and examination parameters.
                float atr{this->atr_parameters[sr.get_param_index()].value()};
                float ex{this->exm_parameters[rank * (rank + 1) / 2 + prev_click_rank[rank]].value()};

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

                this->atr_tmp_parameters[rank * dataset_size + query_index].set_values(new_numerator_atr, 1);
                this->exm_tmp_parameters[(rank * (rank + 1) / 2 + prev_click_rank[rank]) * dataset_size + query_index].set_values(new_numerator_ex, 1);
            }
        }
    };

    // Create threads.
    int n_threads = thread_start_idx.size();
    std::thread threads[n_threads];

    // Divide queries among threads.
    int thread_part = dataset.size() / n_threads;
    int thread_part_left = dataset.size() % n_threads;
    int start_idx{0}, stop_idx{0};

    // Launch threads.
    for (int tid = 0; tid < n_threads; tid++) {
        stop_idx += tid < thread_part_left ? thread_part + 1 : thread_part;
        threads[tid] = std::thread(process_session_thread, std::ref(dataset), tid, start_idx, stop_idx);
        start_idx += tid < thread_part_left ? thread_part + 1 : thread_part;
    }

    // Join threads.
    for (int tid = 0; tid < n_threads; tid++) {
        threads[tid].join();
    }
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
 *
 * @param device Whether to reset the device parameters or the host parameters.
 * (true for device, false for host).
 */
HST void UBM_Hst::reset_parameters(bool device) {
    reset_parameters_hst(this->exm_parameters, this->exm_dptr, device);
    reset_parameters_hst(this->atr_parameters, this->atr_dptr, device);
}

/**
 * @brief Transfers parameters of a given type either from the device to the
 * host, or the otherway around.
 *
 * @param parameter_type The type of parameter that will be transfered.
 * (PUBLIC, PRIVATE, or ALL).
 * @param transfer_direction The direction in which the transfer will happen.
 * (H2D or D2H).
 * @param tmp Whether to transfer the temporary parameters or the originals.
 */
HST void UBM_Hst::transfer_parameters(int parameter_type, int transfer_direction, bool tmp) {
    // Public parameters.
    if (parameter_type == PUBLIC || parameter_type == ALL) {
        if (tmp) transfer_parameters_hst(transfer_direction, this->exm_tmp_parameters, this->exm_tmp_dptr);
        if (!tmp) transfer_parameters_hst(transfer_direction, this->exm_parameters, this->exm_dptr);
    }

    // Private parameters.
    if (parameter_type == PRIVATE || parameter_type == ALL) {
        if (tmp) transfer_parameters_hst(transfer_direction, this->atr_tmp_parameters, this->atr_tmp_dptr);
        if (!tmp) transfer_parameters_hst(transfer_direction, this->atr_parameters, this->atr_dptr);
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
HST void UBM_Hst::get_parameters(std::vector<std::vector<Param>>& destination, int parameter_type) {
    // Add the parameters to a generic vector which can represent  multiple
    // retrieved parameter types.
    if (parameter_type == PUBLIC) {
        destination.resize(1);
        destination[0] = this->exm_parameters;
    }
    else if (parameter_type == PRIVATE) {
        destination.resize(1);
        destination[0] = this->atr_parameters;
    }
    else if (parameter_type == ALL) {
        destination.resize(2);
        destination[0] = this->exm_parameters;
        destination[1] = this->atr_parameters;
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
HST void UBM_Hst::set_parameters(std::vector<std::vector<Param>>& source, int parameter_type) {
    // Set the parameters of this click model.
    if (parameter_type == PUBLIC) {
        this->exm_parameters = source[0];
    }
    else if (parameter_type == PRIVATE) {
        this->atr_parameters = source[0];
    }
    else if (parameter_type == ALL) {
        this->exm_parameters = source[0];
        this->atr_parameters = source[1];
    }
}

/**
 * @brief Get probability of a click on a search result.
 *
 * @param serp The SERP corresponding to a query.
 * @param probabilities The probabilities of a click on each search result.
 */
HST void UBM_Hst::get_serp_probability(SERP_Hst& query_session, float (&probablities)[MAX_SERP]) {
    int prev_click_rank[MAX_SERP] = { 0 };
    query_session.prev_clicked_rank(prev_click_rank);

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        SearchResult_Hst sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        float atr{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1)
            atr = this->atr_parameters[sr.get_param_index()].value();
        float ex{this->exm_parameters[rank * (rank + 1) / 2 + prev_click_rank[rank]].value()};

        // Calculate the click probability.
        probablities[rank] = atr * ex;
    }
}

/**
 * @brief Compute the log-likelihood of the current UBM for the given query
 * session.
 *
 * @param query_session The query session for which the log-likelihood will be
 * computed.
 * @param log_click_probs The vector which will store the log-likelihood for
 * the document at each rank in the query session.
 */
HST void UBM_Hst::get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) {
    int prev_click_rank[MAX_SERP] = { 0 };
    query_session.prev_clicked_rank(prev_click_rank);

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        SearchResult_Hst sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float atr{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1)
            atr = this->atr_parameters[sr.get_param_index()].value();
        float ex{this->exm_parameters[rank * (rank + 1) / 2 + prev_click_rank[rank]].value()};

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
 * @brief Compute the click probability of the current UBM for the given query
 * session.
 *
 * @param query_session The query session for which the click probability will
 * be computed.
 * @param full_click_probs The vector which will store the click probability
 * for the document at each rank in the query session.
 */
HST void UBM_Hst::get_full_click_probs(SERP_Hst& query_session, std::vector<float> &full_click_probs) {
    std::vector<float> temp_full_click_probs;

    // Go through all ranks of the query session.
    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        // Retrieve the search result at the current rank.
        SearchResult_Hst sr = query_session[rank];
        float click_prob{0};

        // Iterate over all previous ranks.
        for (int rank_prev_click{-1}; rank_prev_click < rank; rank_prev_click++) {
            float no_click_between = 1.f;
            int corrected_rank_prev_click{rank_prev_click};

            // Check if this is the first rank. If so, then assign it the highest prev click rank.
            if (rank_prev_click == -1) {
                corrected_rank_prev_click = 9;
            }

            // Search the ranks between the previous click rank and the current document rank.
            for (int rank_between{rank_prev_click + 1}; rank_between < rank; rank_between++) {
                float attr_val{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
                if (query_session[rank_between].get_param_index() != -1)
                    attr_val = this->atr_parameters[query_session[rank_between].get_param_index()].value();

                float exam_val{this->exm_parameters[rank_between * (rank_between + 1) / 2 + corrected_rank_prev_click].value()};

                no_click_between *= 1 - attr_val * exam_val;
            }

            // Get the parameters corresponding to the current search result.
            float attr_val{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
            if (sr.get_param_index() != -1)
                attr_val = this->atr_parameters[query_session[rank].get_param_index()].value();
            float exam_val{this->exm_parameters[rank * (rank + 1) / 2 + corrected_rank_prev_click].value()};
            float temp{no_click_between * (attr_val * exam_val)};

            // Add to the click probability depending on whether a click between was found.
            if (rank_prev_click >= 0) {
                click_prob += temp_full_click_probs[rank_prev_click] * temp;
            } else {
                click_prob += temp;
            }
        }

        temp_full_click_probs.push_back(click_prob);

        // Calculate the full click probability.
        if (sr.get_click() == 1) {
            full_click_probs.push_back(click_prob);
        }
        else {
            full_click_probs.push_back(1 - click_prob);
        }
    }
}

/**
 * @brief Frees the memory allocated to the parameters of this click model on
 * the GPU device.
 */
HST void UBM_Hst::destroy_parameters(void) {
    // Free origin and temporary attractiveness containers.
    CUDA_CHECK(cudaFree(this->atr_dptr));
    CUDA_CHECK(cudaFree(this->atr_tmp_dptr));

    // Free origin and temporary examination containers.
    CUDA_CHECK(cudaFree(this->exm_dptr));
    CUDA_CHECK(cudaFree(this->exm_tmp_dptr));

    // Free the device parameter references and sizes.
    CUDA_CHECK(cudaFree(this->param_refs));
    CUDA_CHECK(cudaFree(this->param_sizes));

    // Reset used device memory.
    this->cm_memory_usage = 0;
}


//---------------------------------------------------------------------------//
// Device-side UBM click model functions.                                    //
//---------------------------------------------------------------------------//

/**
 * @brief Prints a message.
 */
DEV void UBM_Dev::say_hello() {
    printf("Device-side UBM says hello!\n");
}

/**
 * @brief Creates a new UBM click model object.
 *
 * @return The UBM click model object.
 */
DEV UBM_Dev *UBM_Dev::clone() {
    return new UBM_Dev(*this);
}

DEV UBM_Dev::UBM_Dev() = default;

/**
 * @brief Constructs a UBM click model object for the device.
 *
 * @param ubm The base click model object to be copied.
 * @return The UBM click model object.
 */
DEV UBM_Dev::UBM_Dev(UBM_Dev const &ubm) {
}

/**
 * @brief Set the location of the memory allocated for the parameters of this
 * click model on the GPU device.
 *
 * @param parameter_ptr The pointers to the allocated memory.
 * @param parameter_sizes The size of the allocated memory.
 */
DEV void UBM_Dev::set_parameters(Param**& parameter_ptr, int* parameter_sizes) {
    this->atr_parameters = parameter_ptr[0];
    this->atr_tmp_parameters = parameter_ptr[1];
    this->exm_parameters = parameter_ptr[2];
    this->exm_tmp_parameters = parameter_ptr[3];

    this->n_atr_parameters = parameter_sizes[0];
    this->n_atr_tmp_parameters = parameter_sizes[1];
    this->n_exm_parameters = parameter_sizes[2];
    this->n_exm_tmp_parameters = parameter_sizes[3];
}

/**
 * @brief Compute a single Expectation-Maximization iteration for the UBM click
 * model, for a single query session.
 *
 * @param query_session The query session which will be used to estimate the
 * UBM parameters.
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 * @param dataset_size The size of the dataset.
 * @param clicks The click on each rank of the query session.
 * @param pidx The unique parameter index of each rank of the query session.
 */
DEV void UBM_Dev::process_session(SERP_Dev& query_session, int& thread_index, int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    int prev_click_rank[MAX_SERP] = { 0 };
    query_session.prev_clicked_rank(prev_click_rank);

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        // Get the attractiveness and examination parameters.
        float atr{this->atr_parameters[pidx[rank * BLOCK_SIZE + threadIdx.x]].value()};
        float ex{this->exm_parameters[rank * (rank + 1) / 2 + prev_click_rank[rank]].value()};

        // Set the default values of the attractiveness and examination
        // parameters. These will be the parameter values in case the search
        // result document has been clicked.
        float new_numerator_atr{1};
        float new_numerator_ex{1};

        // If the search result document hasn't been clicked, then calculate
        // estimate the parameter value.
        if (clicks[rank * BLOCK_SIZE + threadIdx.x] == 0) {
            // Calculate the current qd-pair click probability.
            float atr_ex = atr * ex;

            // Attractiveness = ((1 - gamma_{r}^{(t)}) * alpha_{qd}^{(t)}) / (1 - (gamma_{r}^{(t)} * alpha_{qd}^{(t)}))
            new_numerator_atr = (atr - atr_ex) / (1 - atr_ex);
            // Examination = ((1 - alpha_{qd}^{(t)}) * gamma_{r}^{(t)}) / (1 - (gamma_{r}^{(t)} * alpha_{qd}^{(t)}))
            new_numerator_ex = (ex - atr_ex) / (1 - atr_ex);
        }

        /** Store the temporary attractiveness and examination parameters.
         * The examination parameters are stored using the following
         * indexing scheme:
         *
         * (rank * (rank + 1) / 2 + previously clicked rank) * dataset size + thread index
         *
         * The scheme creates a 2D list with an increasing number of elements
         * per new entry, as is shown below. These elements are not contiguous,
         * instead, they are spaced apart by a number equal to the number of
         * threads (dataset_size).
         * -------------------------------------
         * rank | prev_click_rank
         * 0    [ 9                            ]
         * 1    [ 0, 9                         ]
         * 2    [ 0, 1, 9                      ]
         * 3    [ 0, 1, 2, 9                   ]
         * 4    [ 0, 1, 2, 3, 9                ]
         * 5    [ 0, 1, 2, 3, 4, 9             ]
         * 6    [ 0, 1, 2, 3, 4, 5, 9          ]
         * 7    [ 0, 1, 2, 3, 4, 5, 6, 9       ]
         * 8    [ 0, 1, 2, 3, 4, 5, 6, 7, 9    ]
         * 9    [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] 55 elements combined
         *
         * An example of mapping the 2D list above to a 1D list is shown below
         * with 3 threads and a SERP_Hst also of length 3.
         * -------------------------------------
         * 1D list   = [ 9, 9, 9, 0, 0, 0, 9, 9, 9, 0, 0, 0, 1, 1, 1, 9, 9, 9 ]
         * Threads   =  T0 T1 T2 T0 T1 T2 T0 T1 T2 T0 T1 T2 T0 T1 T2 T0 T1 T2
         * Rank      =  r0 r0 r0 r1 r1 r1 r1 r1 r1 r2 r2 r2 r2 r2 r2 r2 r2 r2
         * Iteration =   0  0  0  1  1  1  2  2  2  3  3  3  4  4  4  5  5  5
         */
        this->atr_tmp_parameters[rank * dataset_size + thread_index].set_values(new_numerator_atr, 1);
        this->exm_tmp_parameters[(rank * (rank + 1) / 2 + prev_click_rank[rank]) * dataset_size + thread_index].set_values(new_numerator_ex, 1);
    }
}

/**
 * @brief Update the global parameter values using the local parameter values
 * on each thread.
 *
 * @param thread_index The global index of the thread.
 * @param block_index The index of the block in which this thread exists.
 * @param dataset_size The size of the dataset.
 * @param pidx The unique parameter index of each rank of the query session.
 */
DEV void UBM_Dev::update_parameters(int& thread_index, int& block_index, int& dataset_size, const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    update_shared_parameters_dev(this->exm_tmp_parameters, this->exm_parameters, thread_index, this->n_exm_parameters, block_index, dataset_size);

    if (thread_index < dataset_size) {
        update_unique_parameters_dev(this->atr_tmp_parameters, this->atr_parameters, thread_index, dataset_size, pidx);
    }
}
