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
#include <thread>
#include <numeric>
#include <cstring>

// User include.
#include "../utils/definitions.h"
#include "../utils/types.h"
#include "macros.cuh"

HST void get_number_devices(int *num_devices);
HST int get_compute_capability(const int device);
HST void get_device_memory(const int& device_id, size_t& free_memory, size_t& total_memory, const size_t rounding);
HST void get_host_memory(size_t& free_memory, size_t& total_memory, const size_t rounding);
HST DeviceProperties get_device_properties(const int device_id);
HST HostProperties get_host_properties(const int node_id);
HST NodeProperties get_node_properties(const int node_id);
HST void show_help_msg(void);

#endif // CLICK_MODEL_UTILS_H