/** First implementation of a generalized CM base.
 *
 * base.cuh:
 *  - Defines the generalized click model functions.
 */

#ifndef CLICK_MODEL_BASE_H
#define CLICK_MODEL_BASE_H

// System include.
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <cstring>
#include <cuda_runtime.h>

// User include.
#include "../utils/definitions.h"
#include "../utils/types.h"
#include "../utils/macros.cuh"
#include "param.cuh"
#include "../parallel_em/kernel.cuh"

//---------------------------------------------------------------------------//
// Host-side click model.                                                    //
//---------------------------------------------------------------------------//

class ClickModel_Hst {
public:
    HST virtual ClickModel_Hst* clone() = 0;
    HST virtual void say_hello() = 0;

    HST virtual size_t get_memory_usage(void) = 0;
    HST virtual size_t compute_memory_footprint(int n_queries, int n_qd) = 0;
    HST virtual void get_parameter_information(std::pair<std::vector<std::string>, std::vector<std::string>> &headers, std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> &parameters) = 0;
    HST virtual void get_device_references(Param**& param_refs, int*& param_sizes) = 0;

    HST virtual void process_session(const TrainSet& dataset, const std::vector<int>& thread_start_idx) = 0;
    HST virtual void update_parameters(TrainSet& dataset, const std::vector<int>& thread_start_idx) = 0;

    HST virtual void init_parameters(const Partition& dataset, const size_t fmem, const bool device) = 0;
    HST virtual void transfer_parameters(int parameter_type, int transfer_direction, bool tmp = false) = 0;
    HST virtual void get_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) = 0;
    HST virtual void set_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) = 0;
    HST virtual void reset_parameters(bool device) = 0;
    HST virtual void destroy_parameters(void) = 0;

    HST virtual void get_serp_probability(SERP_Hst& query_session, float (&probablities)[MAX_SERP]) = 0;
    HST virtual void get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) = 0;
    HST virtual void get_full_click_probs(SERP_Hst& query_session, std::vector<float> &full_click_probs) = 0;
};

//---------------------------------------------------------------------------//
// Device-side click model.                                                  //
//---------------------------------------------------------------------------//

class ClickModel_Dev {
public:
    DEV virtual ClickModel_Dev* clone() = 0;
    DEV virtual void say_hello() = 0;
    DEV virtual void set_parameters(Param**& parameter_ptr, int* parameter_sizes) = 0;
    DEV virtual void process_session(SERP_Dev& query_session, int& thread_index, int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) = 0;
    DEV virtual void update_parameters(int& thread_index, int& block_index, int& dataset_size, const int (&pidx)[BLOCK_SIZE * MAX_SERP]) = 0;
};

HST ClickModel_Hst* create_cm_host(int model_type);
DEV ClickModel_Dev* create_cm_dev(int model_type);

// Pointer to the device-side click model for all available devices on this node.
DEV static ClickModel_Dev* cm_dev;

#endif // CLICK_MODEL_BASE_H