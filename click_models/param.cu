/** CM parameter.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * param.cu:
 *  - Defines the functions specific to processing the CM parameters.
 */

#include <vector>
#include "param.cuh"

DEV HST Param::Param() = default;

// DEV HST Param::Param(const float& numerator, const float& denominator) {
//     this->numerator = numerator;
//     this->denominator = denominator;
// }

/**
 * @brief Add two parameters.
 *
 * @param other The other parameter to add.
 * @return Param The result of the addition.
 */
DEV HST Param Param::operator + (const Param& other) const {
    Param result;
    result.set_values(this->numerator_val() + other.numerator_val(),
                      this->denominator_val() + other.denominator_val());
    return result;
}

/**
 * @brief Add a value to this parameter.
 *
 * @param value The parameter to be added.
 * @return Param& A reference to this parameter with the added values.
 */
DEV HST Param& Param::operator += (const Param& value) {
    this->numerator += value.numerator_val();
    this->denominator += value.denominator_val();
    return *this;
}

/**
 * @brief Retrieves the click probability of this parameter.
 *
 * @return float The click probability. 0.000001 is returned if the value is
 * too small.
 */
DEV HST float Param::value() const{
    if ((numerator / denominator) < 1.f - 0.000001f) {
        return (numerator / denominator);
    }
    return 1.f - 0.000001f;
}

/**
 * @brief The numerator of this parameter.
 *
 * @return float
 */
DEV HST float Param::numerator_val() const{
    return this->numerator;
}

/**
 * @brief The denominator of this parameter.
 *
 * @return float
 */
DEV HST float Param::denominator_val() const{
    return this->denominator;
}

/**
 * @brief Changes the numerator and denominator of this parameter to the given
 * arguments.
 *
 * @param numerator_val The new numerator value.
 * @param denominator_val The new denominator value.
 */
DEV HST void Param::set_values(float numerator_val, float denominator_val) {
    this->numerator = numerator_val;
    this->denominator = denominator_val;
}

/**
 * @brief Adds the given arguments to the numerator and denominator of this
 * parameter.
 *
 * @param numerator_val The value to add to the parameter numerator.
 * @param denominator_val The value to add to the parameter denominator.
 */
DEV HST void Param::add_to_values(float numerator_val, float denominator_val) {
    this->numerator += numerator_val;
    this->denominator += denominator_val;
}

/**
 * @brief Adds the given arguments to the numerator and denominator of this
 * parameter atomically on the GPU.
 *
 * @param numerator_val The value to add to the parameter numerator.
 * @param denominator_val The value to add to the parameter denominator.
 */
DEV void Param::atomic_add_to_values(float numerator_val, float denominator_val) {
    atomicAddArch(&this->numerator, numerator_val);
    atomicAddArch(&this->denominator, denominator_val);
}
