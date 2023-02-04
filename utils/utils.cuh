/** CUDA utility functions.
 *
 * utils.cuh:
 *  - Defines several host- and device-side (CUDA) utility functions.
 */

#ifndef CLICK_MODEL_UTILS_H
#define CLICK_MODEL_UTILS_H

// System include.
#include <string>
#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>

// User include.
#include "../utils/definitions.h"
#include "macros.cuh"


// DEV void warp_reduce(volatile float* shared_data, int block_index);

HST void get_number_devices(int *num_devices);
HST int get_compute_capability(const int device);
HST int get_warp_size(const int device);
HST void get_device_memory(const int& device_id, size_t& free_memory, size_t& total_memory, const size_t rounding);

HST void show_help_msg(void);
HST void get_host_memory(size_t& free_memory, size_t& total_memory, const size_t rounding);


#endif // CLICK_MODEL_UTILS_H