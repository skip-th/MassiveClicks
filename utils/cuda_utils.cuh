/** First implementation of CUDA utility functions.
 *
 * cuda_utils.cuh:
 *  - Defines several utility functions to prevent code duplication.
 */

#ifndef CLICK_MODEL_CUDA_UTILS_H
#define CLICK_MODEL_CUDA_UTILS_H

// System include.
#include <string>
#include <iostream>

// User include.
#include "../utils/definitions.h"
#include "macros.cuh"

DEV void atomicAddArch(float* address, const float val);

void get_number_devices(int *num_devices);
int get_compute_capability(const int device);
void get_device_memory(const int& device_id, size_t& free_memory, size_t& total_memory, const size_t rounding);

#endif // CLICK_MODEL_CUDA_UTILS_H