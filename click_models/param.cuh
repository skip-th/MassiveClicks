/** CM parameter.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * param.cuh:
 *  - Defines the functions specific to processing the CM parameters.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_PARAMETERS_H
#define CLICK_MODEL_PARAMETERS_H

// System include.
#include <cuda_fp16.h>
#include <math.h>
#include <cuda_runtime.h>

// User include.
#include "../utils/definitions.h"
#include "../utils/utils.cuh"

class Param {
public:
    #ifndef COMPATIBILITY // CUDA >7.0 supports fp16 atomic add
        __half2 fraction{PARAM_DEF_NUM, PARAM_DEF_DENOM};
    #else
        float numerator{PARAM_DEF_NUM};
        float denominator{PARAM_DEF_DENOM};
    #endif

    DEV HST Param();
    DEV HST Param(const float& numerator, const float& denominator);
    DEV HST Param operator + (const Param& other) const;
    DEV HST void operator += (const Param& other);

    DEV HST float value() const;
    DEV HST float numerator_val() const;
    DEV HST float denominator_val() const;
    DEV HST void set_values(float numerator_val, float denominator_val);
    DEV HST void add_to_values(float numerator_val, float denominator_val);
    DEV void atomic_add_to_values(float numerator_val, float denominator_val);
    DEV void atomic_add_param(Param other);
};

#endif // CLICK_MODEL_PARAMETERS_H
