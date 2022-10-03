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


//---------------------------------------------------------------------------//
// Host-side click model functions.                                          //
//---------------------------------------------------------------------------//

class DBN_Hst: public ClickModel_Hst {
public:
    HST DBN_Hst();
    HST DBN_Hst(DBN_Hst const &dbn);
    HST DBN_Hst* clone() override;
    HST void say_hello() override;
    HST size_t get_memory_usage(void) override;
    HST void init_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, const int n_devices) override;
    HST void get_device_references(Param**& param_refs, int*& param_sizes) override;
    HST void update_parameters(int& gridSize, int& blockSize, SERP*& partition, int& dataset_size) override;
    HST void reset_parameters(void) override;

    HST void transfer_parameters(int parameter_type, int transfer_direction) override;
    HST void get_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) override;
    HST void sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters) override;
    HST void set_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) override;
    HST void destroy_parameters(void) override;

    HST void get_log_conditional_click_probs(SERP& query_session, std::vector<float>& log_click_probs) override;
    HST void get_full_click_probs(SERP& search_ses, std::vector<float> &full_click_probs) override;

private:
    HST void init_attractiveness_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, const int n_devices);
    HST void init_satisfaction_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, const int n_devices);
    HST void init_gamma_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, const int n_devices);

    std::vector<Param> attractiveness_parameters; // Host-side attractiveness parameters.
    std::vector<Param> tmp_attractiveness_parameters; // Host-side temporary attractiveness parameters.
    Param* attr_param_dptr; // Pointer to the device-side attractiveness parameters.
    Param* tmp_attr_param_dptr; // Pointer to the device-side temporary attractiveness parameters.
    int n_attr_dev{0}; // Size of the device-side attractiveness parameters.
    int n_tmp_attr_dev{0}; // Size of the device-side temporary attractiveness parameters.

    std::vector<Param> satisfaction_parameters; // Host-side satisfaction parameters.
    std::vector<Param> tmp_satisfaction_parameters; // Host-side temporary satisfaction parameters.
    Param* satisfaction_param_dptr; // Pointer to the device-side satisfaction parameters.
    Param* tmp_satisfaction_param_dptr; // Pointer to the device-side temporary satisfaction parameters.
    int n_satisfaction_dev{0}; // Size of the device-side satisfaction parameters.
    int n_tmp_satisfaction_dev{0}; // Size of the device-side temporary satisfaction parameters.

    std::vector<Param> gamma_parameters; // Host-side continuation parameters.
    std::vector<Param> tmp_gamma_parameters; // Host-side temporary continuation parameters.
    Param* gamma_param_dptr; // Pointer to the device-side continuation parameters.
    Param* tmp_gamma_param_dptr; // Pointer to the device-side temporary continuation parameters.
    int n_gamma_dev{0}; // Size of the device-side continuation parameters.
    int n_tmp_gamma_dev{0}; // Size of the device-side temporary continuation parameters.

    Param** param_refs; // Pointer to the device-side parameter array start pointers.
    int* param_sizes; // Pointer to the device-side sizes of the parameter arrays.

    size_t cm_memory_usage{0}; // Device-side memory usage of the click model parameters.
};


//---------------------------------------------------------------------------//
// Device-side click model functions.                                        //
//---------------------------------------------------------------------------//

class DBN_Dev: public ClickModel_Dev {
public:
    DEV DBN_Dev();
    DEV DBN_Dev(DBN_Dev const &dbn);
    DEV void say_hello() override;
    DEV DBN_Dev* clone() override;
    DEV void set_parameters(Param**& parameter_ptr, int* parameter_sizes) override;
    DEV void process_session(SERP& query_session, int& thread_index, int& partition_size) override;
    DEV void update_parameters(SERP& query_session, int& thread_index, int& block_index, int& partition_size) override;

private:
    DEV void update_examination_parameters(SERP& query_session, int& thread_index, int& block_index, int& partition_size);
    DEV void update_attractiveness_parameters(SERP& query_session, int& thread_index, int& partition_size);
    DEV void update_satisfaction_parameters(SERP& query_session, int& thread_index, int& partition_size);
    DEV void update_gamma_parameters(SERP& query_session, int& thread_index, int& block_index, int& partition_size);

    DEV void compute_exam_car(int& thread_index, SERP& query_session, float (&exam)[MAX_SERP_LENGTH + 1], float (&car)[MAX_SERP_LENGTH + 1]);
    DEV void compute_dbn_attr(int& thread_index, SERP& query_session, int& last_click_rank, float (&exam)[MAX_SERP_LENGTH + 1], float (&car)[MAX_SERP_LENGTH + 1], int& partition_size);
    DEV void compute_dbn_sat(int& thread_index, SERP& query_session, int& last_click_rank, float (&car)[MAX_SERP_LENGTH + 1], int& partition_size);
    DEV void get_tail_clicks(int& thread_index, SERP& query_session, float (&click_probs)[MAX_SERP_LENGTH][MAX_SERP_LENGTH], float (&exam_probs)[MAX_SERP_LENGTH + 1]);
    DEV void compute_gamma(int& thread_index, SERP& query_session, int& last_click_rank, float (&click_probs)[MAX_SERP_LENGTH][MAX_SERP_LENGTH], float (&exam_probs)[MAX_SERP_LENGTH + 1]);

    Param* attractiveness_parameters;
    Param* tmp_attractiveness_parameters;
    int n_attractiveness_parameters{0};
    int n_tmp_attractiveness_parameters{0};

    Param* satisfaction_parameters;
    Param* tmp_satisfaction_parameters;
    int n_satisfaction_parameters{0};
    int n_tmp_satisfaction_parameters{0};

    Param* gamma_parameters;
    Param* tmp_gamma_parameters;
    int n_gamma_parameters{0};
    int n_tmp_gamma_parameters{0};

    int factor_inputs[8][3] = {{0, 0, 0},
                               {0, 0, 1},
                               {0, 1, 0},
                               {0, 1, 1},
                               {1, 0, 0},
                               {1, 0, 1},
                               {1, 1, 0},
                               {1, 1, 1},
    };
};

#endif // CLICK_MODEL_DBN_H