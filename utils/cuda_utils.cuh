/** CUDA utility functions.
 *
 * cuda_utils.cuh:
 *  - Defines several utility functions to prevent code duplication.
 */

#ifndef CLICK_MODEL_CUDA_UTILS_H
#define CLICK_MODEL_CUDA_UTILS_H

// System include.
#include <string>
#include <iostream>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/iterator/transform_iterator.h>
// #include <thrust/iterator/permutation_iterator.h>
// #include <thrust/functional.h>
// #include <thrust/fill.h>
// #include <thrust/device_vector.h>
// #include <thrust/copy.h>

// User include.
#include "../utils/definitions.h"
#include "macros.cuh"

DEV void atomicAddArch(float* address, const float val);

void get_number_devices(int *num_devices);
int get_compute_capability(const int device);
void get_device_memory(const int& device_id, size_t& free_memory, size_t& total_memory, const size_t rounding);

// // The strided range permutation iterator provided by the CUDA Thrust examples
// // repository.
// // https://github.com/NVIDIA/thrust/blob/master/examples/strided_range.cu
// template <typename Iterator>
// class strided_range {
// public:
//     typedef typename thrust::iterator_difference<Iterator>::type difference_type;

//     struct stride_functor : public thrust::unary_function<difference_type,difference_type> {
//         difference_type stride;

//         stride_functor(difference_type stride)
//             : stride(stride) {}

//         __host__ __device__
//         difference_type operator()(const difference_type& i) const {
//             return stride * i;
//         }
//     };

//     typedef typename thrust::counting_iterator<difference_type> CountingIterator;
//     typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
//     typedef typename thrust::permutation_iterator<Iterator,TransformIterator> PermutationIterator;

//     // type of the strided_range iterator
//     typedef PermutationIterator iterator;

//     // construct strided_range for the range [first,last)
//     strided_range(Iterator first, Iterator last, difference_type stride)
//         : first(first), last(last), stride(stride) {}

//     iterator begin(void) const {
//         return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
//     }

//     iterator end(void) const {
//         return begin() + ((last - first) + (stride - 1)) / stride;
//     }

// protected:
//     Iterator first;
//     Iterator last;
//     difference_type stride;
// };

#endif // CLICK_MODEL_CUDA_UTILS_H