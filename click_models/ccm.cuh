/** CCM click model.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * ccm.cuh:
 *  - Defines the functions specific to creating a CCM CM.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_CCM_H
#define CLICK_MODEL_CCM_H

// User include.
#include "../utils/definitions.h"
#include "base.cuh"
#include "factor.cuh"
#include "common.cuh"


//---------------------------------------------------------------------------//
// Host-side click model.                                                    //
//---------------------------------------------------------------------------//

class CCM_Hst: public ClickModel_Hst {
public:
    HST CCM_Hst();
    HST CCM_Hst(CCM_Hst const &ccm);
    HST CCM_Hst* clone() override;
    HST void say_hello() override;

    HST size_t get_memory_usage(void) override;
    HST size_t compute_memory_footprint(int n_queries, int n_qd) override;
    HST void get_device_references(Param**& param_refs, int*& param_sizes) override;

    HST void process_session(const std::vector<SERP_Hst>& dataset, const std::vector<int>& thread_start_idx) override;
    HST void update_parameters(std::vector<SERP_Hst>& dataset, const std::vector<int>& thread_start_idx) override;

    HST void init_parameters(const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& dataset, const size_t fmem, const bool device) override;
    HST void transfer_parameters(int parameter_type, int transfer_direction, bool tmp = false) override;
    HST void get_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) override;
    HST void set_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) override;
    HST void reset_parameters(bool device) override;
    HST void destroy_parameters(void) override;

    HST void get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) override;
    HST void get_full_click_probs(SERP_Hst& search_ses, std::vector<float> &full_click_probs) override;

private:
    HST void compute_exm_car(SERP_Hst& query_session, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1]);
    HST void get_tail_clicks(int& qid, SERP_Hst& query_session, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1]);
    HST void compute_ccm_atr(int& qid, SERP_Hst& query_session, int& last_click_rank, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], int& dataset_size);
    HST void compute_taus(int& qid, SERP_Hst& query_session, int& last_click_rank, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], int& dataset_size);
    HST void compute_tau_1(int& qid, float (&factor_values)[8], float& factor_sum, int& dataset_size);
    HST void compute_tau_2(int& qid, float (&factor_values)[8], float& factor_sum, int& dataset_size);
    HST void compute_tau_3(int& qid, float (&factor_values)[8], float& factor_sum, int& dataset_size);

    HST std::pair<int,int> get_n_atr_params(int n_queries, int n_qd);
    HST std::pair<int,int> get_n_tau_params(int n_queries, int n_qd);

    std::vector<Param> atr_parameters, atr_tmp_parameters; // Host-side attractiveness parameters.
    Param* atr_dptr, *atr_tmp_dptr; // Pointer to the device-side attractiveness parameters.
    int n_atr_params{0}, n_atr_tmp_params{0}; // Size of the attractiveness parameters.

    std::vector<Param> tau_parameters, tau_tmp_parameters; // Host-side continuation parameters.
    Param* tau_dptr, *tau_tmp_dptr; // Pointer to the device-side continuation parameters.
    int n_tau_params{0}, n_tau_tmp_params{0}; // Size of the continuation parameters.

    Param** param_refs; // Pointer to the device-side parameter array start pointers.
    int* param_sizes; // Pointer to the device-side sizes of the parameter arrays.

    size_t cm_memory_usage{0}; // Memory usage of the click model parameters.

    int factor_inputs[8][3] = {{0, 0, 0},
                               {0, 0, 1},
                               {0, 1, 0},
                               {0, 1, 1},
                               {1, 0, 0},
                               {1, 0, 1},
                               {1, 1, 0},
                               {1, 1, 1}};
};


//---------------------------------------------------------------------------//
// Device-side click model.                                                  //
//---------------------------------------------------------------------------//

class CCM_Dev: public ClickModel_Dev {
public:
    DEV CCM_Dev();
    DEV CCM_Dev(CCM_Dev const &ccm);
    DEV void say_hello() override;
    DEV CCM_Dev* clone() override;

    DEV void set_parameters(Param**& parameter_ptr, int* parameter_sizes) override;
    DEV void process_session(SERP_Dev& query_session, int& thread_index, int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) override;
    DEV void update_parameters(int& thread_index, int& block_index, int& dataset_size, const int (&pidx)[BLOCK_SIZE * MAX_SERP]) override;

private:
    DEV void compute_exm_car(float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);
    DEV void get_tail_clicks(float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);
    DEV void compute_ccm_atr(int& thread_index, int& last_click_rank, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);
    DEV void compute_taus(int& thread_index, int& last_click_rank, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);
    DEV void compute_tau_1(int& thread_index, float (&factor_values)[8], float& factor_sum, int& dataset_size);
    DEV void compute_tau_2(int& thread_index, float (&factor_values)[8], float& factor_sum, int& dataset_size);
    DEV void compute_tau_3(int& thread_index, float (&factor_values)[8], float& factor_sum, int& dataset_size);

    Param* atr_parameters, *atr_tmp_parameters; // Device-side attractiveness parameters.
    int n_atr_parameters{0}, n_atr_tmp_parameters{0}; // Size of the attractiveness parameters.

    Param* tau_parameters, *tau_tmp_parameters; // Device-side continuation parameters.
    int n_tau_parameters{0}, n_tau_tmp_parameters{0}; // Size of the continuation parameters.

    int factor_inputs[8][3] = {{0, 0, 0},
                               {0, 0, 1},
                               {0, 1, 0},
                               {0, 1, 1},
                               {1, 0, 0},
                               {1, 0, 1},
                               {1, 1, 0},
                               {1, 1, 1}};
};

#endif // CLICK_MODEL_CCM_H