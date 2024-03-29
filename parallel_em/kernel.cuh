/** CUDA kernel functions.
 *
 * kernel.cuh:
 *  - Defines several CUDA kernels usable when training a click model.
 */

#ifndef CLICK_MODEL_KERNEL_H
#define CLICK_MODEL_KERNEL_H

// User include.
#include "../utils/definitions.h"
#include "../utils/macros.cuh"
#include "../utils/utils.cuh"
#include "../click_models/base.cuh"
#include "../click_models/param.cuh"
#include "../data/search.cuh"

namespace Kernel {
    GLB void initialize(const int model_type, const int node_id, const int device_id, Param** cm_param_ptr, int* parameter_sizes);
    GLB void em_training(SearchResult_Dev* partition, int partition_size);
    GLB void update(SearchResult_Dev* partition, int partition_size);
}

#endif // CLICK_MODEL_KERNEL_H