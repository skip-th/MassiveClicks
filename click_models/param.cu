/** CM parameter.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * param.cu:
 *  - Defines the functions specific to processing the CM parameters.
 */

#include "param.cuh"

DEV HST Param::Param() = default;

DEV HST Param::Param(const float& numerator, const float& denominator) {
    #ifndef COMPATIBILITY
        this->fraction = __floats2half2_rn(numerator, denominator);
    #else
        this->numerator = numerator;
        this->denominator = denominator;
    #endif
}

/**
 * @brief Add two parameters.
 *
 * @param other The other parameter to add.
 * @return Param The result of the addition.
 */
DEV HST Param Param::operator + (const Param& other) const {
    Param result;

    #ifndef COMPATIBILITY
        #ifdef __CUDA_ARCH__ // Use CUDA intrinsics when on the GPU
            result.fraction = __hfma2(this->fraction, __floats2half2_rn(0.5f, 0.5f), other.fraction);
        #else
            result.fraction = __floats2half2_rn(__half2float(this->fraction.x) + __half2float(other.fraction.x),
                                                __half2float(this->fraction.y) + __half2float(other.fraction.y));
        #endif
    #else
        result.numerator = this->numerator + other.numerator;
        result.denominator = this->denominator + other.denominator;
    #endif

    return result;
}

/**
 * @brief Add a value to this parameter.
 *
 * @param value The parameter to be added.
 * @return Param& A reference to this parameter with the added values.
 */
DEV HST void Param::operator += (const Param& other) {
    #ifndef COMPATIBILITY
        #ifdef __CUDA_ARCH__ // Use CUDA intrinsics when on the GPU
            this->fraction = __hfma2(this->fraction, __floats2half2_rn(0.5f, 0.5f), other.fraction);
        #else
            this->fraction = __floats2half2_rn(__half2float(this->fraction.x) + __half2float(other.fraction.x),
                                               __half2float(this->fraction.y) + __half2float(other.fraction.y));
        #endif
    #else
        this->numerator += other.numerator;
        this->denominator += other.denominator;
    #endif
}


/**
 * @brief Retrieves the click probability of this parameter.
 *
 * @return float The click probability. 0.000001 is returned if the value is
 * too small.
 */
DEV HST float Param::value() const {
    #ifndef COMPATIBILITY
        #ifdef __CUDA_ARCH__
            if (__hlt(__hdiv(this->fraction.x, this->fraction.y), 1.f - 0.000001f)) {
                return __hdiv(this->fraction.x, this->fraction.y);
            }
        #else
            if ((__half2float(this->fraction.x) / __half2float(this->fraction.y)) < 1.f - 0.000001f) {
                return __half2float(this->fraction.x) / __half2float(this->fraction.y);
            }
        #endif
    #else
        if ((this->numerator / this->denominator) < 1.f - 0.000001f) {
            return (this->numerator / this->denominator);
        }
    #endif
    return 1.f - 0.000001f;
}


/**
 * @brief The numerator of this parameter.
 *
 * @return float The numerator.
 */
DEV HST float Param::numerator_val() const{
    #ifndef COMPATIBILITY
        return 	__half2float(this->fraction.x);
    #else
        return this->numerator;
    #endif
}

/**
 * @brief The denominator of this parameter.
 *
 * @return float The denominator.
 */
DEV HST float Param::denominator_val() const{
    #ifndef COMPATIBILITY
        return 	__half2float(this->fraction.y);
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
    #ifndef COMPATIBILITY
        this->fraction = __floats2half2_rn(numerator_val, denominator_val);
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
    #ifndef COMPATIBILITY
        #ifdef __CUDA_ARCH__ // Use CUDA intrinsics when on the GPU
            this->fraction = __hadd2(this->fraction, __floats2half2_rn(numerator_val, denominator_val));
        #else
            this->fraction = __floats2half2_rn(__half2float(this->fraction.x) + numerator_val,
                                               __half2float(this->fraction.y) + denominator_val);
        #endif
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
    #ifndef COMPATIBILITY
        atomicAdd(&this->fraction, __floats2half2_rn(numerator_val, denominator_val));
    #else
        atomicAdd(&this->numerator, numerator_val);
        atomicAdd(&this->denominator, denominator_val);
    #endif
}

/**
 * @brief Adds the given arguments to the numerator and denominator of this
 * parameter atomically on the GPU.
 *
 * @param numerator_val The value to add to the parameter numerator.
 * @param denominator_val The value to add to the parameter denominator.
 */
DEV void Param::atomic_add_param(Param other) {
    #ifndef COMPATIBILITY
        atomicAdd(&this->fraction, other.fraction);
    #else
        atomicAdd(&this->numerator, other.numerator_val());
        atomicAdd(&this->denominator, other.denominator_val());
    #endif
}