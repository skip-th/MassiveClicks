/** Click model similar commonly used functions.
 *
 * common.cuh:
 *  - Defines functions for updating host-side parameters.
 *  - Defines functions for updating device-side parameters.
 *  - Defines function for allocating parameter memory.
 *  - Defines function for reseting parameter memory.
 *  - Defines function transfering parameters between host and device.
 */

#ifndef CLICK_MODEL_COMMON_H
#define CLICK_MODEL_COMMON_H

#include "../utils/definitions.h"
#include "base.cuh"
#include "../parallel_em/communicator.h"

HST void init_parameters_hst(std::vector<Param>& params, std::vector<Param>& params_tmp, Param*& param_dptr, Param*& param_tmp_dptr, std::pair<int, int> n_params, int& n_params_dev, int& n_tmp_params_dev,
                             size_t& cm_memory_usage, const std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>& dataset, const size_t fmem, const bool device);
HST void reset_parameters_hst(std::vector<Param>& params, Param* dptr, bool device);
HST void transfer_parameters_hst(int transfer_direction, std::vector<Param>& params, Param* dptr);
HST void update_unique_parameters_hst(std::vector<Param>& src, std::vector<Param>& dst, const std::vector<SERP_Hst>& dataset, const std::vector<int>& thread_start_idx);
HST void update_shared_parameters_hst(std::vector<Param>& src, std::vector<Param>& dst, const std::vector<SERP_Hst>& dataset, const std::vector<int>& thread_start_idx);
DEV void update_shared_parameters_dev(Param*& src, Param*& dst, int& thread_index, int& src_size, int& block_index, int& dataset_size);
DEV void update_unique_parameters_dev(Param*& src, Param*& dst, int& thread_index, int& dataset_size, const int (&pidx)[BLOCK_SIZE * MAX_SERP]);

#endif // CLICK_MODEL_COMMON_H