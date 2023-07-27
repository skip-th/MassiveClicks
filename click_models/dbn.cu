/** DBN click model.
 *
 * dbn.cu:
 *  - Defines the functions specific to creating a DBN CM.
 */

#include "dbn.cuh"


//---------------------------------------------------------------------------//
// Host-side DBN click model functions.                                      //
//---------------------------------------------------------------------------//

HST DBN_Hst::DBN_Hst() = default;

/**
 * @brief Constructs a DBN click model object for the host.
 *
 * @param dbn The base click model object to copy.
 * @return The DBN click model object.
 */
HST DBN_Hst::DBN_Hst(DBN_Hst const &dbn) {
}

/**
 * @brief Creates a new DBN click model object.
 *
 * @return The DBN click model object.
 */
HST DBN_Hst* DBN_Hst::clone() {
    return new DBN_Hst(*this);
}

/**
 * @brief Print a message.
 */
HST void DBN_Hst::say_hello() {
    std::cout << "Host-side DBN says hello!" << std::endl;
}

/**
 * @brief Get the amount of device memory allocated to this click model.
 *
 * @return The used memory.
 */
HST size_t DBN_Hst::get_memory_usage(void) {
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
HST size_t DBN_Hst::compute_memory_footprint(int n_queries, int n_qd) {
    std::pair<int, int> n_attractiveness = this->get_n_atr_params(n_queries, n_qd);
    std::pair<int, int> n_satisfaction = this->get_n_sat_params(n_queries, n_qd);
    std::pair<int, int> n_continuation = this->get_n_gam_params(n_queries, n_qd);

    return (n_attractiveness.first + n_attractiveness.second +
            n_satisfaction.first + n_satisfaction.second +
            n_continuation.first + n_continuation.second) * sizeof(Param);
}

/**
 * @brief Get the number of original and temporary attractiveness parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return The number of original and temporary examination
 * parameters.
 */
HST std::pair<int,int> DBN_Hst::get_n_atr_params(int n_queries, int n_qd) {
    return std::make_pair(n_qd,                  // # original
                          n_queries * MAX_SERP); // # temporary
}

/**
 * @brief Get the number of original and temporary satisfaction parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return The number of original and temporary satisfaction
 * parameters.
 */
HST std::pair<int, int> DBN_Hst::get_n_sat_params(int n_queries, int n_qd) {
    return std::make_pair(n_qd,                  // # original
                          n_queries * MAX_SERP); // # temporary
}

/**
 * @brief Get the number of original and temporary continuation (gamma) parameters.
 *
 * @param n_queries The number of queries assigned to this click model.
 * @param n_qd The number of query-document pairs assigned to this click model.
 * @return The number of original and temporary continuation
 * parameters.
 */
HST std::pair<int, int> DBN_Hst::get_n_gam_params(int n_queries, int n_qd) {
    return std::make_pair(N_GAM,              // # original
                          n_queries * N_GAM); // # temporary
}

/**
 * @brief Allocate device-side memory for the attractiveness, satisfaction and
 * continuation parameters of the click model.
 *
 * @param dataset The training and testing sets, and the number of
 * query-document pairs in the training set.
 * @param n_devices The number of devices on this node.
 * @param fmem The amount of free memory on the device.
 * @param device The device to allocate memory on.
 */
HST void DBN_Hst::init_parameters(const Partition& dataset, const size_t fmem, const bool device) {
    std::pair<int, int> n_attractiveness = this->get_n_atr_params(std::get<0>(dataset).size(), std::get<2>(dataset));
    init_parameters_hst(this->atr_parameters, this->atr_tmp_parameters, this->atr_dptr, this->atr_tmp_dptr, n_attractiveness, this->n_atr_params, this->n_atr_tmp_params, this->cm_memory_usage, dataset, fmem, device);
    std::pair<int, int> n_satisfaction = this->get_n_sat_params(std::get<0>(dataset).size(), std::get<2>(dataset));
    init_parameters_hst(this->sat_parameters, this->sat_tmp_parameters, this->sat_dptr, this->sat_tmp_dptr, n_satisfaction, this->n_sat_params, this->n_sat_tmp_params, this->cm_memory_usage, dataset, fmem, device);
    std::pair<int, int> n_continuation = this->get_n_gam_params(std::get<0>(dataset).size(), std::get<2>(dataset));
    init_parameters_hst(this->gam_parameters, this->gam_tmp_parameters, this->gam_dptr, this->gam_tmp_dptr, n_continuation, this->n_gam_params, this->n_gam_tmp_params, this->cm_memory_usage, dataset, fmem, device);
}

/**
 * @brief Get the name of the parameters of this click model.
 *
 * @return The public and private parameter names.
 */
HST void DBN_Hst::get_parameter_information(
        std::pair<std::vector<std::string>, std::vector<std::string>> &headers,
        std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> &parameters) {
    // Set parameter headers.
    std::vector<std::string> public_name = {"continuation"};
    std::vector<std::string> private_name = {"attractiveness", "satisfaction"};
    headers = std::make_pair(public_name, private_name);

    // Set parameter values.
    std::vector<std::vector<Param> *> public_parameters = {&this->gam_parameters};
    std::vector<std::vector<Param> *> private_parameters = {&this->atr_parameters, &this->sat_parameters};
    parameters = std::make_pair(public_parameters, private_parameters);
}

/**
 * @brief Get the references to the allocated device-side memory.
 *
 * @param param_refs An array containing the references to the device-side
 * parameters in memory.
 * @param param_sizes The size of each of the memory allocations on the device.
 */
HST void DBN_Hst::get_device_references(Param**& param_refs, int*& param_sizes) {
    int n_references = 6;

    // Create a temporary array to store the device references.
    Param* tmp_param_refs_array[n_references];
    tmp_param_refs_array[0] = this->atr_dptr;
    tmp_param_refs_array[1] = this->atr_tmp_dptr;
    tmp_param_refs_array[2] = this->sat_dptr;
    tmp_param_refs_array[3] = this->sat_tmp_dptr;
    tmp_param_refs_array[4] = this->gam_dptr;
    tmp_param_refs_array[5] = this->gam_tmp_dptr;

    // Allocate space for the device references.
    CUDA_CHECK(cudaMalloc(&param_refs, n_references * sizeof(Param*)));
    CUDA_CHECK(cudaMemcpy(param_refs, tmp_param_refs_array,
                          n_references * sizeof(Param*), cudaMemcpyHostToDevice));

    int tmp_param_sizes_array[n_references];
    tmp_param_sizes_array[0] = this->n_atr_params;
    tmp_param_sizes_array[1] = this->n_atr_tmp_params;
    tmp_param_sizes_array[2] = this->n_sat_params;
    tmp_param_sizes_array[3] = this->n_sat_tmp_params;
    tmp_param_sizes_array[4] = this->n_gam_params;
    tmp_param_sizes_array[5] = this->n_gam_tmp_params;

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
HST void DBN_Hst::update_parameters(TrainSet& dataset, const std::vector<int>& thread_start_idx) {
    update_unique_parameters_hst(this->atr_tmp_parameters, this->atr_parameters, dataset, thread_start_idx);
    update_unique_parameters_hst(this->sat_tmp_parameters, this->sat_parameters, dataset, thread_start_idx);
    update_shared_parameters_hst(this->gam_tmp_parameters, this->gam_parameters, dataset, thread_start_idx);
}

/**
 * @brief Compute a single Expectation-Maximization iteration for the DBN click
 * model for each query session.
 *
 * @param dataset The training set.
 * @param thread_start_idx Dataset starting indices of each thread.
 */
HST void DBN_Hst::process_session(const TrainSet& dataset, const std::vector<int>& thread_start_idx) {
    // Iterate over the queries in the dataset in each thread.
    auto process_session_thread = [this](const TrainSet& dataset, const int thread_idx, int start_idx, int stop_idx) {
        int dataset_size = dataset.size();

        for (int query_index = start_idx; query_index < stop_idx; query_index++) {
            // Retrieve the search results associated with the current query.
            SERP_Hst query_session = dataset[query_index];

            int last_click_rank = query_session.last_click_rank();
            float click_probs[MAX_SERP][MAX_SERP] = { 0.f };
            float exam_probs[MAX_SERP + 1];
            float exam[MAX_SERP + 1];
            float car[MAX_SERP + 1] = { 0.f };

            this->gam_tmp_parameters[query_index].set_values(0.f, 0.f);

            this->compute_exm_car(query_session, exam, car);
            this->compute_dbn_atr(query_index, query_session, last_click_rank, exam, car, dataset_size);
            this->compute_dbn_sat(query_index, query_session, last_click_rank, car, dataset_size);
            this->get_tail_clicks(query_index, query_session, click_probs, exam_probs);
            this->compute_gamma(query_index, query_session, last_click_rank, click_probs, exam_probs);
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
        threads[tid] = std::thread(process_session_thread, dataset, tid, start_idx, stop_idx);
        start_idx += tid < thread_part_left ? thread_part + 1 : thread_part;
    }

    // Join threads.
    for (int tid = 0; tid < n_threads; tid++) {
        threads[tid].join();
    }
}

/**
 * @brief Compute the examination parameter for every rank of this query
 * session. The examination parameter can be re-computed every iteration using
 * the values from attractiveness, satisfaction, and continuation parameters
 * from the previous iteration.
 *
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param exam The examination parameters for every rank. The first rank is
 * always examined (1).
 * @param car
 */
HST void DBN_Hst::compute_exm_car(SERP_Hst& query_session, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1]) {
    // Set the default examination value for the first rank.
    exam[0] = 1.f;

    float attr_val, sat_value, gamma_value, ex_value, temp, car_val;
    float car_helper[MAX_SERP][2];

    for (int rank = 0; rank < MAX_SERP;) {
        SearchResult_Hst sr = query_session[rank];

        attr_val = this->atr_parameters[sr.get_param_index()].value();
        sat_value = this->sat_parameters[sr.get_param_index()].value();
        gamma_value = this->gam_parameters[0].value();
        ex_value = exam[rank];

        temp = gamma_value * (1 - attr_val);
        ex_value *= temp + gamma_value * attr_val * (1 - sat_value);

        car_helper[rank][0] = attr_val;
        car_helper[rank][1] = temp;

        rank += 1;
        exam[rank] = ex_value;
    }

    for (int car_itr = MAX_SERP - 1; car_itr > -1; car_itr--) {
        car_val = car[car_itr + 1];

        car[car_itr] = car_helper[car_itr][0] + car_helper[car_itr][1] * car_val;
    }
}

/**
 * @brief Compute the attractiveness parameter for every rank of this query
 * session.
 *
 * @param qid The index of the query session in the dataset.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param exam The examination parameters for every rank. The first rank is
 * always examined (1).
 * @param car
 * @param dataset_size The size of the dataset.
 */
HST void DBN_Hst::compute_dbn_atr(int& qid, SERP_Hst& query_session, int& last_click_rank, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], int& dataset_size) {
    float numerator_update, denominator_update;
    float exam_val, attr_val,  car_val;

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        SearchResult_Hst sr = query_session[rank];

        numerator_update = 0.f;
        denominator_update = 1.f;

        if (sr.get_click() == 1) {
            numerator_update += 1.f;
        }
        else if (rank >= last_click_rank) {
            attr_val = this->atr_parameters[sr.get_param_index()].value();
            exam_val = exam[rank];
            car_val = car[rank];

            numerator_update += (attr_val * (1 - exam_val)) / (1 - exam_val * car_val);
        }

        this->atr_tmp_parameters[rank * dataset_size + qid].set_values(numerator_update, denominator_update);
    }
}

/**
 * @brief Compute the satisfaction parameter for every rank of this query
 * session.
 *
 * @param qid The index of the query session in the dataset.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param car
 * @param dataset_size The size of the dataset.
 */
HST void DBN_Hst::compute_dbn_sat(int& qid, SERP_Hst& query_session, int& last_click_rank, float (&car)[MAX_SERP + 1], int& dataset_size) {
    float numerator_update, denominator_update;
    float gamma_val, sat_val, car_val;

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        SearchResult_Hst sr = query_session[rank];

        if (sr.get_click() == 1) {
            numerator_update = 0.f;
            denominator_update = 1.f;

            if (rank == last_click_rank) {
                sat_val = this->sat_parameters[sr.get_param_index()].value();
                gamma_val = this->gam_parameters[0].value();

                if (rank < MAX_SERP - 1) {
                    car_val = car[rank + 1];
                } else{
                    car_val = 0.f;
                }

                numerator_update += sat_val / (1 - (1 - sat_val) * gamma_val * car_val);
            }

            this->sat_tmp_parameters[rank * dataset_size + qid].set_values(numerator_update, denominator_update);
        }
    }
}

/**
 * @brief Compute the click probabilities of a rank given the clicks on the
 * preceding ranks.
 *
 * @param qid The index of the query session in the dataset.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param click_probs The probabilty of a click occurring on a rank.
 * @param exam_probs The probability of a rank being examined.
 */
HST void DBN_Hst::get_tail_clicks(int& qid, SERP_Hst& query_session, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1]) {
    exam_probs[0] = 1.f;
    float exam_val, gamma_val, click_prob;

    for (int start_rank = 0; start_rank < MAX_SERP; start_rank++) {
        exam_val = 1.f;

        int ses_itr{0};
        for (int res_itr = start_rank; res_itr < MAX_SERP; res_itr++) {
            SearchResult_Hst tmp_sr = query_session[ses_itr];

            float attr_val = this->atr_parameters[tmp_sr.get_param_index()].value();
            float sat_val = this->sat_parameters[tmp_sr.get_param_index()].value();
            gamma_val = this->gam_parameters[0].value();

            if (query_session[res_itr].get_click() == 1) {
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
 * @param qid The index of the query session in the dataset.
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param click_probs The probabilty of a click occurring on a rank.
 * @param exam_probs The probability of a rank being examined.
 */
HST void DBN_Hst::compute_gamma(int& qid, SERP_Hst& query_session, int& last_click_rank, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1]) {
    float factor_values[8] = { 0.f };

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++){
        SearchResult_Hst sr = query_session[rank];

        // Send the initialization values to the phi function.
        DBNFactor factor_func(click_probs, exam_probs, sr.get_click(),
                              last_click_rank, rank,
                              this->atr_parameters[sr.get_param_index()].value(),
                              this->sat_parameters[sr.get_param_index()].value(),
                              this->gam_parameters[0].value());

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

        this->gam_tmp_parameters[qid].add_to_values(numerator_update, denominator_update);
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
HST void DBN_Hst::reset_parameters(bool device) {
    reset_parameters_hst(this->sat_parameters, this->sat_dptr, device);
    reset_parameters_hst(this->atr_parameters, this->atr_dptr, device);
    reset_parameters_hst(this->gam_parameters, this->gam_dptr, device);
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
HST void DBN_Hst::transfer_parameters(int parameter_type, int transfer_direction, bool tmp) {
    // Public parameters.
    if (parameter_type == PUBLIC || parameter_type == ALL) {
        if (tmp) transfer_parameters_hst(transfer_direction, this->gam_tmp_parameters, this->gam_tmp_dptr);
        if (!tmp) transfer_parameters_hst(transfer_direction, this->gam_parameters, this->gam_dptr);
    }

    // Private parameters.
    if (parameter_type == PRIVATE || parameter_type == ALL) {
        if (tmp) transfer_parameters_hst(transfer_direction, this->sat_tmp_parameters, this->sat_tmp_dptr);
        if (!tmp) transfer_parameters_hst(transfer_direction, this->sat_parameters, this->sat_dptr);
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
HST void DBN_Hst::get_parameters(std::vector<std::vector<Param>>& destination, int parameter_type) {
    // Add the parameters to a generic vector which can represent  multiple
    // retrieved parameter types.
    if (parameter_type == PUBLIC) {
        destination.resize(1);
        destination[0] = this->gam_parameters;
    }
    else if (parameter_type == PRIVATE) {
        destination.resize(2);
        destination[0] = this->atr_parameters;
        destination[1] = this->sat_parameters;
    }
    else if (parameter_type == ALL) {
        destination.resize(3);
        destination[0] = this->atr_parameters;
        destination[1] = this->sat_parameters;
        destination[2] = this->gam_parameters;
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
HST void DBN_Hst::set_parameters(std::vector<std::vector<Param>>& source, int parameter_type) {
    // Set the parameters of this click model.
    if (parameter_type == PUBLIC) {
        this->gam_parameters = source[0];
    }
    else if (parameter_type == PRIVATE) {
        this->atr_parameters = source[0];
        this->sat_parameters = source[1];
    }
    else if (parameter_type == ALL) {
        this->atr_parameters = source[0];
        this->sat_parameters = source[1];
        this->gam_parameters = source[2];
    }
}

/**
 * @brief Get probability of a click on a search result.
 *
 * @param query_session The query session containing the search results.
 * @param probabilities The probabilities of a click on each search result.
 */
HST void DBN_Hst::get_serp_probability(SERP_Hst& query_session, float (&probablities)[MAX_SERP]) {
    float ex{1.f}, click_prob;

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        SearchResult_Hst sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float attr_val{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        float sat_val{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1) {
            attr_val = this->atr_parameters[sr.get_param_index()].value();
            sat_val = this->sat_parameters[sr.get_param_index()].value();
        }
        float gamma_val{this->gam_parameters[0].value()};

        if (sr.get_click() == 1) {
            click_prob = attr_val * ex;
            ex = gamma_val * ( 1- sat_val);
        } else{
            click_prob = 1 - attr_val * ex;
            ex *= gamma_val * ( 1 - attr_val) / click_prob;
        }
        // Calculate the click probability.
        probablities[rank] = click_prob;
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
HST void DBN_Hst::get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) {
    float ex{1.f}, click_prob;

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        SearchResult_Hst sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float attr_val{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        float sat_val{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1) {
            attr_val = this->atr_parameters[sr.get_param_index()].value();
            sat_val = this->sat_parameters[sr.get_param_index()].value();
        }
        float gamma_val{this->gam_parameters[0].value()};

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
HST void DBN_Hst::get_full_click_probs(SERP_Hst& query_session, std::vector<float> &full_click_probs) {
    float ex{1.f}, atr_mul_ex;

    // Go through all ranks of the query session.
    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        // Retrieve the search result at the current rank.
        SearchResult_Hst sr = query_session[rank];

        // Get the parameters corresponding to the current search result.
        // Return the default parameter value if the qd-pair was not found in
        // the training set.
        float atr{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        float sat{(float) PARAM_DEF_NUM / (float) PARAM_DEF_DENOM};
        if (sr.get_param_index() != -1) {
            atr = this->atr_parameters[sr.get_param_index()].value();
            sat = this->sat_parameters[sr.get_param_index()].value();
        }
        float gamma{this->gam_parameters[0].value()};

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
HST void DBN_Hst::destroy_parameters(void) {
    // Free origin and temporary attractiveness containers.
    CUDA_CHECK(cudaFree(this->atr_dptr));
    CUDA_CHECK(cudaFree(this->atr_tmp_dptr));

    // Free origin and temporary satisfaction containers.
    CUDA_CHECK(cudaFree(this->sat_dptr));
    CUDA_CHECK(cudaFree(this->sat_tmp_dptr));

    // Free origin and temporary continuation containers.
    CUDA_CHECK(cudaFree(this->gam_dptr));
    CUDA_CHECK(cudaFree(this->gam_tmp_dptr));

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
 * @return The DBN click model object.
 */
DEV DBN_Dev *DBN_Dev::clone() {
    return new DBN_Dev(*this);
}

DEV DBN_Dev::DBN_Dev() = default;

/**
 * @brief Constructs a DBN click model object for the device.
 *
 * @param dbn The base click model object to be copied.
 * @return The DBN click model object.
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
    this->atr_parameters = parameter_ptr[0];
    this->atr_tmp_parameters = parameter_ptr[1];
    this->sat_parameters = parameter_ptr[2];
    this->sat_tmp_parameters = parameter_ptr[3];
    this->gam_parameters = parameter_ptr[4];
    this->gam_tmp_parameters = parameter_ptr[5];

    // Set parameter array sizes.
    this->n_atr_parameters = parameter_sizes[0];
    this->n_atr_tmp_parameters = parameter_sizes[1];
    this->n_sat_parameters = parameter_sizes[2];
    this->n_sat_tmp_parameters = parameter_sizes[3];
    this->n_gam_parameters = parameter_sizes[4];
    this->n_gam_tmp_parameters = parameter_sizes[5];
}

/**
 * @brief Compute a single Expectation-Maximization iteration for the DBN click
 * model, for a single query session.
 *
 * @param query_session The query session which will be used to estimate the
 * DBN parameters.
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 * @param dataset_size The size of the dataset.
 * @param clicks The click on each rank of the query session.
 * @param pidx The parameter index of each rank of the query session.
 */
DEV void DBN_Dev::process_session(SERP_Dev& query_session, int& thread_index, int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    int last_click_rank = query_session.last_click_rank();
    float click_probs[MAX_SERP][MAX_SERP] = { 0.f };
    float exam_probs[MAX_SERP + 1];
    float exam[MAX_SERP + 1];
    float car[MAX_SERP + 1] = { 0.f };

    this->gam_tmp_parameters[thread_index].set_values(0.f, 0.f);

    this->compute_exm_car(exam, car, pidx);
    this->compute_dbn_atr(thread_index, last_click_rank, exam, car, dataset_size, clicks, pidx);
    this->compute_dbn_sat(thread_index, last_click_rank, car, dataset_size, clicks, pidx);
    this->get_tail_clicks(click_probs, exam_probs, clicks, pidx);
    this->compute_gamma(thread_index, last_click_rank, click_probs, exam_probs, clicks, pidx);
}

/**
 * @brief Compute the examination parameter for every rank of this query
 * session. The examination parameter can be re-computed every iteration using
 * the values from attractiveness, satisfaction, and continuation parameters
 * from the previous iteration.
 *
 * @param exam The examination parameters for every rank. The first rank is
 * always examined (1).
 * @param car
 * @param pidx The parameter index of each rank of the query session.
 */
DEV void DBN_Dev::compute_exm_car(float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    // Set the default examination value for the first rank.
    exam[0] = 1.f;

    float attr_val, sat_value, gamma_value, ex_value, temp, car_val;
    float car_helper[MAX_SERP][2];
    int shr_idx;

    for (int rank = 0; rank < MAX_SERP;) {
        shr_idx = pidx[rank * BLOCK_SIZE + threadIdx.x];
        attr_val = this->atr_parameters[shr_idx].value();
        sat_value = this->sat_parameters[shr_idx].value();
        gamma_value = this->gam_parameters[0].value();
        ex_value = exam[rank];

        temp = gamma_value * (1 - attr_val);
        ex_value *= temp + gamma_value * attr_val * (1 - sat_value);

        car_helper[rank][0] = attr_val;
        car_helper[rank][1] = temp;

        rank += 1;
        exam[rank] = ex_value;
    }

    for (int car_itr = MAX_SERP - 1; car_itr > -1; car_itr--) {
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
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param exam The examination parameters for every rank. The first rank is
 * always examined (1).
 * @param car
 * @param dataset_size The size of the dataset.
 * @param clicks The click on each rank of the query session.
 */
DEV void DBN_Dev::compute_dbn_atr(int& thread_index, int& last_click_rank, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    float numerator_update, denominator_update;
    float exam_val, attr_val,  car_val;

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        numerator_update = 0.f;
        denominator_update = 1.f;

        if (clicks[rank * BLOCK_SIZE + threadIdx.x] == 1) {
            numerator_update += 1.f;
        }
        else if (rank >= last_click_rank) {
            attr_val = this->atr_parameters[pidx[rank * BLOCK_SIZE + threadIdx.x]].value();
            exam_val = exam[rank];
            car_val = car[rank];

            numerator_update += (attr_val * (1 - exam_val)) / (1 - exam_val * car_val);
        }

        this->atr_tmp_parameters[rank * dataset_size + thread_index].set_values(numerator_update, denominator_update);
    }
}

/**
 * @brief Compute the satisfaction parameter for every rank of this query
 * session.
 *
 * @param thread_index The index of the thread which will be estimating the
 * parameters.
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param car
 * @param dataset_size The size of the dataset.
 * @param clicks The click on each rank of the query session.
 * @param pidx The parameter index of each rank of the query session.
 */
DEV void DBN_Dev::compute_dbn_sat(int& thread_index, int& last_click_rank, float (&car)[MAX_SERP + 1], int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    float numerator_update, denominator_update;
    float gamma_val, sat_val, car_val;

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        if (clicks[rank * BLOCK_SIZE + threadIdx.x] == 1) {
            numerator_update = 0.f;
            denominator_update = 1.f;

            if (rank == last_click_rank) {
                sat_val = this->sat_parameters[pidx[rank * BLOCK_SIZE + threadIdx.x]].value();
                gamma_val = this->gam_parameters[0].value();

                if (rank < MAX_SERP - 1) {
                    car_val = car[rank + 1];
                } else{
                    car_val = 0.f;
                }

                numerator_update += sat_val / (1 - (1 - sat_val) * gamma_val * car_val);
            }

            this->sat_tmp_parameters[rank * dataset_size + thread_index].set_values(numerator_update, denominator_update);
        }
    }
}

/**
 * @brief Compute the click probabilities of a rank given the clicks on the
 * preceding ranks.
 *
 * @param click_probs The probabilty of a click occurring on a rank.
 * @param exam_probs The probability of a rank being examined.
 * @param clicks The click on each rank of the query session.
 * @param pidx The parameter index of each rank of the query session.
 */
DEV void DBN_Dev::get_tail_clicks(float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    exam_probs[0] = 1.f;
    float exam_val, gamma_val, click_prob;

    for (int start_rank = 0; start_rank < MAX_SERP; start_rank++) {
        exam_val = 1.f;

        int ses_itr{0};
        for (int res_itr = start_rank; res_itr < MAX_SERP; res_itr++) {
            int shr_idx = pidx[ses_itr * BLOCK_SIZE + threadIdx.x];
            float attr_val = this->atr_parameters[shr_idx].value();
            float sat_val = this->sat_parameters[shr_idx].value();
            gamma_val = this->gam_parameters[0].value();

            if (clicks[res_itr * BLOCK_SIZE + threadIdx.x] == 1){
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
 * @param last_click_rank The last rank of this query sessions which has been
 * clicked.
 * @param click_probs The probabilty of a click occurring on a rank.
 * @param exam_probs The probability of a rank being examined.
 * @param clicks The click on each rank of the query session.
 * @param pidx The parameter index of each rank of the query session.
 */
DEV void DBN_Dev::compute_gamma(int& thread_index, int& last_click_rank, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    float factor_values[8] = { 0.f };

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++){
        // Send the initialization values to the phi function.
        int shr_idx = pidx[rank * BLOCK_SIZE + threadIdx.x];
        DBNFactor factor_func(click_probs, exam_probs, clicks[rank * BLOCK_SIZE + threadIdx.x],
                              last_click_rank, rank,
                              this->atr_parameters[shr_idx].value(),
                              this->sat_parameters[shr_idx].value(),
                              this->gam_parameters[0].value());

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

        this->gam_tmp_parameters[thread_index].add_to_values(numerator_update, denominator_update);
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
DEV void DBN_Dev::update_parameters(int& thread_index, int& block_index, int& dataset_size, const int (&pidx)[BLOCK_SIZE * MAX_SERP]) {
    update_shared_parameters_dev(this->gam_tmp_parameters, this->gam_parameters, thread_index, this->n_gam_parameters, block_index, dataset_size);

    if (thread_index < dataset_size) {
        update_unique_parameters_dev(this->atr_tmp_parameters, this->atr_parameters, thread_index, dataset_size, pidx);
        update_unique_parameters_dev(this->sat_tmp_parameters, this->sat_parameters, thread_index, dataset_size, pidx);
    }
}
