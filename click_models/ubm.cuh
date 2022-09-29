/** UBM click model.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * ubm.cuh:
 *  - Defines the functions specific to creating a UBM CM.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_UBM_H
#define CLICK_MODEL_UBM_H

// User include.
#include "../utils/definitions.h"
#include "base.cuh"


//---------------------------------------------------------------------------//
// Host-side click model functions.                                          //
//---------------------------------------------------------------------------//

class UBM_Hst: public ClickModel_Hst {
public:
    HST UBM_Hst();
    HST UBM_Hst(UBM_Hst const &ubm);
    HST UBM_Hst* clone() override;
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
    HST void init_examination_parameters(const std::tuple<std::vector<SERP>, std::vector<SERP>, int>& partition, const int n_devices);

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

class UBM_Dev: public ClickModel_Dev {
public:
    DEV UBM_Dev();
    DEV UBM_Dev(UBM_Dev const &ubm);
    DEV void say_hello() override;
    DEV UBM_Dev* clone() override;
    DEV void set_parameters(Param**& parameter_ptr, int* parameter_sizes) override; //, int*& parameter_sizes)  override;
    DEV void process_session(SERP& query_session, int& thread_index) override;
    DEV void update_parameters(SERP& query_session, int& thread_index, int& block_index, int& partition_size) override;

private:
    DEV void update_examination_parameters(SERP& query_session, int& thread_index, int& block_index, int& partition_size);
    DEV void update_attractiveness_parameters(SERP& query_session, int& thread_index);

    Param* attractiveness_parameters;
    Param* tmp_attractiveness_parameters;
    int n_attractiveness_parameters{0};
    int n_tmp_attractiveness_parameters{0};

    Param* examination_parameters;
    Param* tmp_examination_parameters;
    int n_examination_parameters{0};
    int n_tmp_examination_parameters{0};
};

#endif // CLICK_MODEL_UBM_H