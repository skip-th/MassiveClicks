/** PBM click model.
 *
 * pbm.cuh:
 *  - Defines the functions specific to creating a PBM CM.
 */

#ifndef CLICK_MODEL_PBM_H
#define CLICK_MODEL_PBM_H

// User include.
#include "../utils/definitions.h"
#include "base.cuh"
#include "common.cuh"


//---------------------------------------------------------------------------//
// Host-side click model.                                                    //
//---------------------------------------------------------------------------//

class PBM_Hst: public ClickModel_Hst {
public:
    HST PBM_Hst();
    HST PBM_Hst(PBM_Hst const &pbm);
    HST PBM_Hst* clone() override;
    HST void say_hello() override;

    HST size_t get_memory_usage(void) override;
    HST size_t compute_memory_footprint(int n_queries, int n_qd) override;
    HST void get_parameter_information(std::pair<std::vector<std::string>, std::vector<std::string>> &headers, std::pair<std::vector<std::vector<Param> *>, std::vector<std::vector<Param> *>> &parameters) override;
    HST void get_device_references(Param**& param_refs, int*& param_sizes) override;

    HST void process_session(const TrainSet& dataset, const std::vector<int>& thread_start_idx) override;
    HST void update_parameters(TrainSet& dataset, const std::vector<int>& thread_start_idx) override;

    HST void init_parameters(const Partition& dataset, const size_t fmem, const bool device) override;
    HST void transfer_parameters(int parameter_type, int transfer_direction, bool tmp = false) override;
    HST void get_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) override;
    HST void set_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) override;
    HST void reset_parameters(bool device) override;
    HST void destroy_parameters(void) override;

    HST void get_serp_probability(SERP_Hst& query_session, float (&probablities)[MAX_SERP]) override;
    HST void get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) override;
    HST void get_full_click_probs(SERP_Hst& query_session, std::vector<float>& full_click_probs) override;

private:
    HST std::pair<int,int> get_n_atr_params(int n_queries, int n_qd);
    HST std::pair<int,int> get_n_exm_params(int n_queries, int n_qd);

    std::vector<Param> atr_parameters, atr_tmp_parameters; // Host-side attractiveness parameters.
    Param* atr_dptr, *atr_tmp_dptr; // Pointer to the device-side attractiveness parameters.
    int n_atr_params{0}, n_atr_tmp_params{0}; // Size of the attractiveness parameters.

    std::vector<Param> exm_parameters, exm_tmp_parameters; // Host-side examination parameters.
    Param* exm_dptr, *exm_tmp_dptr; // Pointer to the device-side examination parameters.
    int n_exm_params{0}, n_exm_tmp_params{0}; // Size of the attractiveness parameters.

    Param** param_refs; // Pointer to the device-side parameter array start pointers.
    int* param_sizes; // Pointer to the device-side sizes of the parameter arrays.

    size_t cm_memory_usage{0}; // Memory usage of the click model parameters.
};


//---------------------------------------------------------------------------//
// Device-side click model.                                                  //
//---------------------------------------------------------------------------//

class PBM_Dev: public ClickModel_Dev {
public:
    DEV PBM_Dev();
    DEV PBM_Dev(PBM_Dev const &pbm);
    DEV void say_hello() override;
    DEV PBM_Dev* clone() override;

    DEV void set_parameters(Param**& parameter_ptr, int* parameter_sizes) override;
    DEV void process_session(SERP_Dev& query_session, int& thread_index, int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) override;
    DEV void update_parameters(int& thread_index, int& block_index, int& dataset_size, const int (&pidx)[BLOCK_SIZE * MAX_SERP]) override;

private:
    Param* atr_parameters, *atr_tmp_parameters; // Host-side attractiveness parameters.
    int n_atr_parameters{0}, n_atr_tmp_parameters{0}; // Size of the attractiveness parameters.

    Param* exm_parameters, *exm_tmp_parameters; // Host-side examination parameters.
    int n_exm_parameters{0}, n_exm_tmp_parameters{0}; // Size of the examination parameters.
};

#endif // CLICK_MODEL_PBM_H