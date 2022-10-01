/** CM parameter.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * param.cuh:
 *  - Defines the functions specific to processing the CM parameters.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_PARAMETERS_H
#define CLICK_MODEL_PARAMETERS_H

// User include.
#include "../utils/definitions.h"
#include "../utils/cuda_utils.cuh"


class Param {
private:
    float numerator{PARAM_DEF_NUM};
    float denominator{PARAM_DEF_DENOM};
public:
    DEV HST Param();
    DEV HST Param(const float& numerator, const float& denominator);
    // DEV HST Param operator + (const Param &other) const;
    // DEV HST Param& operator = (const Param& value);
    // DEV HST Param& operator += (const Param& value);
    DEV HST float value() const;
    DEV HST float numerator_val() const;
    DEV HST float denominator_val() const;
    DEV HST void set_values(float numerator_val, float denominator_val);
    DEV HST void add_to_values(float numerator_val, float denominator_val);
    DEV void atomic_add_to_values(float numerator_val, float denominator_val);
};

// struct Param_add: public thrust::binary_function<Param, Param, Param> {
//     DEV Param operator()(Param a, Param b) { return a + b; }
// };

#endif // CLICK_MODEL_PARAMETERS_H
