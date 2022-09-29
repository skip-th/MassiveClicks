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

// User include.
#include "../utils/definitions.h"
#include "../utils/macros.cuh"
#include "../data/dataset.h"
#include "param.cuh"
#include "../parallel_em/kernel.cuh"


//---------------------------------------------------------------------------//
// Host-side click model functions.                                          //
//---------------------------------------------------------------------------//

class ClickModel_Host {
public:
    // Functions are virtual, so they can be overloaded by a derived class, like PBM_Host.
    HST virtual ClickModel_Host* clone() = 0;
    HST virtual void say_hello() = 0;
    HST virtual size_t get_memory_usage(void) = 0;
    HST virtual void init_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, int n_devices) = 0;
    HST virtual void get_device_references(Param**& param_refs, int*& param_sizes) = 0;
    HST virtual void update_parameters(int& gridSize, int& blockSize, SERP*& partition, int& dataset_size) = 0;
    HST virtual void reset_parameters(void) = 0;

    HST virtual void transfer_parameters(int parameter_type, int transfer_direction) = 0;
    HST virtual void get_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) = 0;
    HST virtual void sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters) = 0;
    HST virtual void set_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) = 0;
    HST virtual void destroy_parameters(void) = 0;

    HST virtual void get_log_conditional_click_probs(SERP& query_session, std::vector<float>& log_click_probs) = 0;
    HST virtual void get_full_click_probs(SERP& search_ses, std::vector<float> &full_click_probs) = 0;
};


//---------------------------------------------------------------------------//
// Device-side click model functions.                                        //
//---------------------------------------------------------------------------//

class ClickModel_Dev {
public:
    // Functions are virtual, so they can be overloaded by a derived class, like PBM_Dev.
    DEV virtual ClickModel_Dev* clone() = 0;
    DEV virtual void say_hello() = 0;
    DEV virtual void set_parameters(Param**& parameter_ptr, int* parameter_sizes) = 0;
    DEV virtual void process_session(SERP& query_session, int& thread_index) = 0;
    DEV virtual void update_parameters(SERP& query_session, int& thread_index, int& block_index, int& parameter_type, int& partition_size) = 0;
};

HST ClickModel_Host* create_cm_host(int model_type);
DEV ClickModel_Dev* create_cm_dev(int model_type);

// Pointer to the device-side click model for all available devices on this node.
DEV static ClickModel_Dev* cm_dev;


#endif // CLICK_MODEL_BASE_H