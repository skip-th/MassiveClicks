/** CUDA utility functions.
 *
 * cuda_utils.cu:
 *  - Defines several utility functions to prevent code duplication.
 */

#include "cuda_utils.cuh"

// Source: Cuda Toolkit Documentation, B.14. Atomic Functions.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if __CUDA_ARCH__ >= 600
    /**
     * @brief Atomically adds a float value to another float at a given
     * address for CUDA architecture >6.0.
     *
     * @param address The address to add the value to.
     * @param val The value to add.
     */
    DEV void atomicAddArch(float* address, const float val) {
        atomicAdd_system(address, val);
    }
#else
    /**
     * @brief Atomically adds a float value to another float at a given
     * address for CUDA architecture <6.0.
     *
     * @param address The address to add the value to.
     * @param val The value to add.
     */
    DEV void atomicAddArch(float* address, const float val) {
        unsigned int* address_as_ull = (unsigned int*) address;
        unsigned int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __float_as_uint(val + __uint_as_float(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
    }
#endif

/**
 * @brief Get the number of GPU devices available on this machine.
 *
 * @param num_devices Integer used to store the number of GPU devices.
 */
void get_number_devices(int *num_devices) {
    CUDA_CHECK(cudaGetDeviceCount(num_devices));
}

/**
 * @brief Get the compute capability of a specific GPU device.
 *
 * @param device The ID of the GPU device.
 * @return int The compute capability of the GPU device (e.g. 5.2 = 52).
 */
int get_compute_capability(const int device) {
    cudaDeviceProp dprop;
    CUDA_CHECK(cudaGetDeviceProperties(&dprop, device));

    return std::stoi(std::to_string(dprop.major) + std::to_string(dprop.minor));
}

/**
 * @brief Get the memory usage of a specific GPU device.
 *
 * @param device_id The ID of the GPU device.
 * @param free_memory The free memory of the GPU device.
 * @param total_memory The total memory of the GPU device.
 * @param rounding The number by which the free and total memory will be rounded.
 */
void get_device_memory(const int& device_id, size_t& free_memory, size_t& total_memory, const size_t rounding) {
    int old_device;
    CUDA_CHECK(cudaGetDevice(&old_device));
    CUDA_CHECK(cudaSetDevice(device_id));

    size_t free_bytes, total_bytes;

    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    free_memory = free_bytes / rounding;
    total_memory = total_bytes / rounding;

    CUDA_CHECK(cudaSetDevice(old_device));
}
