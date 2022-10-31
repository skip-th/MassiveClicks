/** First implementation of a generalized CM base.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * base.cuh:
 *  - Defines the generalized click model functions.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_BASE_H
#define CLICK_MODEL_BASE_H

// System include.
#include <iostream>
#include <algorithm>
// #include <mutex>
#include <pthread.h>
#include <cstring>

// User include.
#include "../utils/definitions.h"
#include "../utils/macros.cuh"
#include "../data/dataset.h"
#include "param.cuh"
#include "../parallel_em/kernel.cuh"

//---------------------------------------------------------------------------//
// Host-side click model functions.                                          //
//---------------------------------------------------------------------------//

class ClickModel_Hst {
public:
    HST virtual ClickModel_Hst* clone() = 0;
    HST virtual void say_hello() = 0;
    HST virtual size_t get_memory_usage(void) = 0;
    HST virtual size_t compute_memory_footprint(int n_queries, int n_qd) = 0;
    HST virtual void init_parameters(const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition, const size_t fmem) = 0;
    HST virtual void get_device_references(Param**& param_refs, int*& param_sizes) = 0;
    HST virtual void update_parameters(int& gridSize, int& blockSize, SERP_Dev*& partition, int& dataset_size) = 0;
    HST virtual void update_parameters_on_host(const std::vector<int>& thread_start_idx, std::vector<SERP_Hst>& partition) = 0;
    HST virtual void reset_parameters(void) = 0;

    HST virtual void transfer_parameters(int parameter_type, int transfer_direction) = 0;
    HST virtual void get_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) = 0;
    HST virtual void sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters) = 0;
    HST virtual void set_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) = 0;
    HST virtual void destroy_parameters(void) = 0;

    HST virtual void get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) = 0;
    HST virtual void get_full_click_probs(SERP_Hst& search_ses, std::vector<float> &full_click_probs) = 0;
};

//---------------------------------------------------------------------------//
// Device-side click model functions.                                        //
//---------------------------------------------------------------------------//

class ClickModel_Dev {
public:
    DEV virtual ClickModel_Dev* clone() = 0;
    DEV virtual void say_hello() = 0;
    DEV virtual void set_parameters(Param**& parameter_ptr, int* parameter_sizes) = 0;
    DEV virtual void process_session(SERP_Dev& query_session, int& thread_index, int& partition_size) = 0;
    DEV virtual void update_parameters(SERP_Dev& query_session, int& thread_index, int& block_index, int& partition_size) = 0;
};

HST ClickModel_Hst* create_cm_host(int model_type);
DEV ClickModel_Dev* create_cm_dev(int model_type);

// Pointer to the device-side click model for all available devices on this node.
DEV static ClickModel_Dev* cm_dev;


#endif // CLICK_MODEL_BASE_H