/** CM parameter.
 *
 * param.cuh:
 *  - Defines the functions specific to processing the CM parameters.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_PARAMETERS_H
#define CLICK_MODEL_PARAMETERS_H

// System include.
#include <math.h>
#include <cuda_runtime.h>

// User include.
#include "../utils/definitions.h"
#include "../utils/utils.cuh"

class Param {
private:
    #ifdef __CUDA_ARCH__
        float2 fraction{PARAM_DEF_NUM, PARAM_DEF_DENOM};
    #else
        float numerator{PARAM_DEF_NUM};
        float denominator{PARAM_DEF_DENOM};
    #endif
public:
    DEV Param(const struct float2& fraction);
    DEV HST Param();
    DEV HST Param(const float& numerator, const float& denominator);
    DEV HST Param operator + (const Param& other) const;
    DEV HST Param& operator += (const Param& value);

    DEV HST float value() const;
    DEV HST float numerator_val() const;
    DEV HST float denominator_val() const;
    DEV HST void set_values(float numerator_val, float denominator_val);
    DEV HST void add_to_values(float numerator_val, float denominator_val);
    DEV void atomic_add_to_values(float numerator_val, float denominator_val);
};

#endif // CLICK_MODEL_PARAMETERS_H
