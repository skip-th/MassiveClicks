/** CUDA utility functions.
 *
 * utils.cuh:
 *  - Defines several host- and device-side (CUDA) utility functions.
 *  - Defines several macros for error handling.
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
#include <mpi.h>

// User include.
#include "../utils/definitions.h"
#include "../utils/types.h"
#include "macros.cuh"

// MPI error handling macro.
#define MPI_CHECK(call) \
    if ((call) != MPI_SUCCESS) { \
        std::cerr << "MPI error calling \""#call"\"\n"; \
        std::cout << "Quiting MPI" << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, -1); }

// CUDA error handling macro.
#define CUDA_CHECK(call) \
    if ((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        std::cerr << "CUDA error " << err << ": " << cudaGetErrorName(err) << \
        ".\n\tAt " << __FILE__ << ":" << __LINE__ << " in function \"" << \
        __func__ << "\".\n\tCall: \""#call"\"" << ".\n\tDescription: " << \
        cudaGetErrorString(err) << "." << std::endl; \
        std::cout << "Quiting MPI" << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, err); }

// Conditional print stream to print only with the root node.
class ConditionalStream {
    private:
        const ProcessingConfig& config;
        const int src;
        std::ostream& out;

    public:
        ConditionalStream(const ProcessingConfig& cfg, const int s, std::ostream& os = std::cout)
            : config(cfg), src(s), out(os) {}

        template<typename T>
        ConditionalStream& operator<<(const T& value) {
            if (config.node_id == src) {
                out << value;
            }
            return *this;
        }

        ConditionalStream& operator<<(std::ostream& (*func)(std::ostream&)) {
            if (config.node_id == src) {
                out << func;
            }
            return *this;
        }
    };

HST void get_number_devices(int *num_devices);
HST int get_compute_capability(const int device);
HST void get_device_memory(const int& device_id, size_t& free_memory, size_t& total_memory, const size_t rounding);
HST void get_host_memory(size_t& free_memory, size_t& total_memory, const size_t rounding);
HST DeviceProperties get_device_properties(const int device_id);
HST HostProperties get_host_properties(const int node_id);
HST NodeProperties get_node_properties(const int node_id);
HST void show_help_msg(void);

#endif // CLICK_MODEL_UTILS_H