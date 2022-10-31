/** PBM click model.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * pbm.cuh:
 *  - Defines the functions specific to creating a PBM CM.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_PBM_H
#define CLICK_MODEL_PBM_H

// User include.
#include "../utils/definitions.h"
#include "base.cuh"


//---------------------------------------------------------------------------//
// Host-side click model functions.                                          //
//---------------------------------------------------------------------------//

class PBM_Hst: public ClickModel_Hst {
public:
    HST PBM_Hst();
    HST PBM_Hst(PBM_Hst const &pbm);
    HST PBM_Hst* clone() override;
    HST void say_hello() override;
    HST size_t get_memory_usage(void) override;
    HST size_t compute_memory_footprint(int n_queries, int n_qd) override;
    HST void init_parameters(const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition, const size_t fmem) override;
    HST void get_device_references(Param**& param_refs, int*& param_sizes) override;
    HST void update_parameters(int& gridSize, int& blockSize, SERP_Dev*& partition, int& dataset_size) override;
    HST void update_parameters_on_host(const std::vector<int>& thread_start_idx, std::vector<SERP_Hst>& partition) override;
    HST void reset_parameters(void) override;

    HST void transfer_parameters(int parameter_type, int transfer_direction) override;
    HST void get_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) override;
    HST void sync_parameters(std::vector<std::vector<std::vector<Param>>>& parameters) override;
    HST void set_parameters(std::vector<std::vector<Param>>& public_parameters, int parameter_type) override;
    HST void destroy_parameters(void) override;

    HST void get_log_conditional_click_probs(SERP_Hst& query_session, std::vector<float>& log_click_probs) override;
    HST void get_full_click_probs(SERP_Hst& search_ses, std::vector<float>& full_click_probs) override;

private:
    HST void init_attractiveness_parameters(const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition, const size_t fmem);
    HST void init_examination_parameters(const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& partition, const size_t fmem);
    HST void* update_examination_parameters(void* args);
    HST void* update_attractiveness_parameters(void* args);
    HST static void* update_ex_init(void* args) { return ((PBM_Hst*)args)->update_examination_parameters(args); }
    HST static void* update_attr_init(void* args) { return ((PBM_Hst*)args)->update_attractiveness_parameters(args); }
    HST std::pair<int,int> get_n_attr_params(int n_queries, int n_qd);
    HST std::pair<int,int> get_n_exam_params(int n_queries, int n_qd);

    std::vector<Param> attractiveness_parameters; // Host-side attractiveness parameters.
    std::vector<Param> tmp_attractiveness_parameters; // Host-side temporary attractiveness parameters.
    Param* attr_param_dptr; // Pointer to the device-side attractiveness parameters.
    Param* tmp_attr_param_dptr; // Pointer to the device-side temporary attractiveness parameters.
    int n_attr_dev{0}; // Size of the device-side attractiveness parameters.
    int n_tmp_attr_dev{0}; // Size of the device-side temporary attractiveness parameters.

    std::vector<Param> examination_parameters; // Host-side examination parameters.
    std::vector<Param> tmp_examination_parameters; // Host-side temporary examination parameters.
    Param* exam_param_dptr; // Pointer to the device-side examination parameters.
    Param* tmp_exam_param_dptr; // Pointer to the device-side temporary examination parameters.
    int n_exams_dev{0}; // Size of the device-side attractiveness parameters.
    int n_tmp_exams_dev{0}; // Size of the device-side temporary attractiveness parameters.

    Param** param_refs; // Pointer to the device-side parameter array start pointers.
    int* param_sizes; // Pointer to the device-side sizes of the parameter arrays.

    size_t cm_memory_usage{0}; // Device-side memory usage of the click model parameters.
};


//---------------------------------------------------------------------------//
// Device-side click model functions.                                        //
//---------------------------------------------------------------------------//

class PBM_Dev: public ClickModel_Dev {
public:
    DEV PBM_Dev();
    DEV PBM_Dev(PBM_Dev const &pbm);
    DEV void say_hello() override;
    DEV PBM_Dev* clone() override;
    DEV void set_parameters(Param**& parameter_ptr, int* parameter_sizes) override; //, int*& parameter_sizes)  override;
    DEV void process_session(SERP_Dev& query_session, int& thread_index, int& partition_size) override;
    DEV void update_parameters(SERP_Dev& query_session, int& thread_index, int& block_index, int& partition_size) override;

private:
    DEV void update_examination_parameters(SERP_Dev& query_session, int& thread_index, int& block_index, int& partition_size);
    DEV void update_attractiveness_parameters(SERP_Dev& query_session, int& thread_index, int& partition_size);

    Param* attractiveness_parameters;
    Param* tmp_attractiveness_parameters;
    int n_attractiveness_parameters{0};
    int n_tmp_attractiveness_parameters{0};

    Param* examination_parameters;
    Param* tmp_examination_parameters;
    int n_examination_parameters{0};
    int n_tmp_examination_parameters{0};
};

#endif // CLICK_MODEL_PBM_H