/** DBN click model.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * dbn.cuh:
 *  - Defines the functions specific to creating a DBN CM.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_DBN_H
#define CLICK_MODEL_DBN_H

// User include.
#include "../utils/definitions.h"
#include "base.cuh"
#include "factor.cuh"
#include "common.cuh"


//---------------------------------------------------------------------------//
// Host-side click model.                                                    //
//---------------------------------------------------------------------------//

class DBN_Hst: public ClickModel_Hst {
public:
    HST DBN_Hst();
    HST DBN_Hst(DBN_Hst const &dbn);
    HST DBN_Hst* clone() override;
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

    HST void get_serp_probability(SERP_Hst& query_session, float (&probablities)[MAX_SERP]) override;
    HST void get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) override;
    HST void get_full_click_probs(SERP_Hst& query_session, std::vector<float> &full_click_probs) override;

private:
    HST void compute_exm_car(SERP_Hst& query_session, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1]);
    HST void compute_dbn_atr(int& qid, SERP_Hst& query_session, int& last_click_rank, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], int& dataset_size);
    HST void compute_dbn_sat(int& qid, SERP_Hst& query_session, int& last_click_rank, float (&car)[MAX_SERP + 1], int& dataset_size);
    HST void get_tail_clicks(int& qid, SERP_Hst& query_session, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1]);
    HST void compute_gamma(int& qid, SERP_Hst& query_session, int& last_click_rank, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1]);

    HST std::pair<int,int> get_n_atr_params(int n_queries, int n_qd);
    HST std::pair<int,int> get_n_sat_params(int n_queries, int n_qd);
    HST std::pair<int,int> get_n_gam_params(int n_queries, int n_qd);

    std::vector<Param> atr_parameters, atr_tmp_parameters; // Host-side attractiveness parameters.
    Param* atr_dptr, *atr_tmp_dptr; // Pointer to the device-side attractiveness parameters.
    int n_atr_params{0}, n_atr_tmp_params{0}; // Size of the attractiveness parameters.

    std::vector<Param> sat_parameters, sat_tmp_parameters; // Host-side satisfaction parameters.
    Param* sat_dptr, *sat_tmp_dptr; // Pointer to the device-side satisfaction parameters.
    int n_sat_params{0}, n_sat_tmp_params{0}; // Size of the satisfaction parameters.

    std::vector<Param> gam_parameters, gam_tmp_parameters; // Host-side continuation parameters.
    Param* gam_dptr, *gam_tmp_dptr; // Pointer to the device-side continuation parameters.
    int n_gam_params{0}, n_gam_tmp_params{0}; // Size of the continuation parameters.

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

class DBN_Dev: public ClickModel_Dev {
public:
    DEV DBN_Dev();
    DEV DBN_Dev(DBN_Dev const &dbn);
    DEV void say_hello() override;
    DEV DBN_Dev* clone() override;

    DEV void set_parameters(Param**& parameter_ptr, int* parameter_sizes) override;
    DEV void process_session(SERP_Dev& query_session, int& thread_index, int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]) override;
    DEV void update_parameters(int& thread_index, int& block_index, int& dataset_size, const int (&pidx)[BLOCK_SIZE * MAX_SERP]) override;

private:
    DEV void compute_exm_car(float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);
    DEV void compute_dbn_atr(int& thread_index, int& last_click_rank, float (&exam)[MAX_SERP + 1], float (&car)[MAX_SERP + 1], int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);
    DEV void compute_dbn_sat(int& thread_index, int& last_click_rank, float (&car)[MAX_SERP + 1], int& dataset_size, const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);
    DEV void get_tail_clicks(float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);
    DEV void compute_gamma(int& thread_index, int& last_click_rank, float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], const char (&clicks)[BLOCK_SIZE * MAX_SERP], const int (&pidx)[BLOCK_SIZE * MAX_SERP]);

    Param* atr_parameters, *atr_tmp_parameters; // Device-side attractiveness parameters.
    int n_atr_parameters{0}, n_atr_tmp_parameters{0}; // Size of the attractiveness parameters.

    Param* sat_parameters, *sat_tmp_parameters; // Device-side satisfaction parameters.
    int n_sat_parameters{0}, n_sat_tmp_parameters{0}; // Size of the satisfaction parameters.

    Param* gam_parameters, *gam_tmp_parameters; // Device-side continuation parameters.
    int n_gam_parameters{0}, n_gam_tmp_parameters{0}; // Size of the continuation parameters.

    int factor_inputs[8][3] = {{0, 0, 0},
                               {0, 0, 1},
                               {0, 1, 0},
                               {0, 1, 1},
                               {1, 0, 0},
                               {1, 0, 1},
                               {1, 1, 0},
                               {1, 1, 1}};
};

#endif // CLICK_MODEL_DBN_H