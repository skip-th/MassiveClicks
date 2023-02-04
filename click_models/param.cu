/** CM parameter.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * param.cu:
 *  - Defines the functions specific to processing the CM parameters.
 */

#include "param.cuh"

DEV HST Param::Param() = default;

DEV HST Param::Param(const float& numerator, const float& denominator) {
    #ifdef __CUDA_ARCH__
        this->fraction = make_float2(numerator, denominator);
        // this->fraction.x = numerator_val;
        // this->fraction.y = denominator_val;
    #else
        this->numerator = numerator;
        this->denominator = denominator;
    #endif
}

DEV Param::Param(const struct float2& fraction) {
    #ifdef __CUDA_ARCH__
        this->fraction = fraction;
    #endif
}

/**
 * @brief Add two parameters.
 *
 * @param other The other parameter to add.
 * @return The result of the addition.
 */
DEV HST Param Param::operator + (const Param& other) const {
    Param result;
    #ifdef __CUDA_ARCH__
        result.set_values(this->fraction.x + other.numerator_val(),
                          this->fraction.y + other.denominator_val());
        // this->fraction = make_float2(numerator, denominator);
        // this->fraction.x = numerator_val;
        // this->fraction.y = denominator_val;
    #else
        result.set_values(this->numerator_val() + other.numerator_val(),
                          this->denominator_val() + other.denominator_val());
        // this->numerator = numerator;
        // this->denominator = denominator;
    #endif
    return result;
}

/**
 * @brief Add a value to this parameter.
 *
 * @param value The parameter to be added.
 * @return A reference to this parameter with the added values.
 */
DEV HST Param& Param::operator += (const Param& value) {
    #ifdef __CUDA_ARCH__
        // this->fraction += __float2float2_rn(numerator_val, denominator_val);
        this->fraction.x += value.numerator_val();
        this->fraction.y += value.denominator_val();
    #else
        this->numerator += value.numerator_val();
        this->denominator += value.denominator_val();
    #endif
    return *this;
}

/**
 * @brief Retrieves the click probability of this parameter.
 *
 * @return The click probability. 0.000001 is returned if the value is
 * too small.
 */
DEV HST float Param::value() const{
    #ifdef __CUDA_ARCH__
        if ((this->fraction.x / this->fraction.y) < 1.f - 0.000001f) {
            return (this->fraction.x / this->fraction.y);
        }
    #else
        if ((numerator / denominator) < 1.f - 0.000001f) {
            return (numerator / denominator);
        }
    #endif

    return 1.f - 0.000001f;
}

/**
 * @brief The numerator of this parameter.
 *
 * @return The numerator.
 */
DEV HST float Param::numerator_val() const{
    #ifdef __CUDA_ARCH__
        return this->fraction.x;
    #else
        return this->numerator;
    #endif
}

/**
 * @brief The denominator of this parameter.
 *
 * @return The denominator.
 */
DEV HST float Param::denominator_val() const{
    #ifdef __CUDA_ARCH__
        return this->fraction.y;
    #else
        return this->denominator;
    #endif
}

/**
 * @brief Changes the numerator and denominator of this parameter to the given
 * arguments.
 *
 * @param numerator_val The new numerator value.
 * @param denominator_val The new denominator value.
 */
DEV HST void Param::set_values(float numerator_val, float denominator_val) {
    #ifdef __CUDA_ARCH__
        this->fraction = make_float2(numerator_val, denominator_val);
    #else
        this->numerator = numerator_val;
        this->denominator = denominator_val;
    #endif
}

/**
 * @brief Adds the given arguments to the numerator and denominator of this
 * parameter.
 *
 * @param numerator_val The value to add to the parameter numerator.
 * @param denominator_val The value to add to the parameter denominator.
 */
DEV HST void Param::add_to_values(float numerator_val, float denominator_val) {
    #ifdef __CUDA_ARCH__
        this->fraction.x += numerator_val;
        this->fraction.y += denominator_val;
    #else
        this->numerator += numerator_val;
        this->denominator += denominator_val;
    #endif
}

/**
 * @brief Adds the given arguments to the numerator and denominator of this
 * parameter atomically on the GPU.
 *
 * @param numerator_val The value to add to the parameter numerator.
 * @param denominator_val The value to add to the parameter denominator.
 */
DEV void Param::atomic_add_to_values(float numerator_val, float denominator_val) {
    #ifdef __CUDA_ARCH__
        atomicAdd(&this->fraction.x, numerator_val);
        atomicAdd(&this->fraction.y, denominator_val);
    #endif
}
