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
     * @brief Atomically adds a single-precision floating point value to
     * another at a given address for CUDA architecture of 6.0 and above.
     *
     * @param address The address to add the value to.
     * @param val The value to add.
     */
    DEV void atomicAddArch(float* address, const float val) {
        atomicAdd_system(address, val);
    }
#else
    /**
     * @brief Atomically adds a single-precision floating point value to
     * another at a given address for CUDA architecture below 6.0.
     *
     * @param address The address to add the value to.
     * @param val The value to add.
     */
    DEV void atomicAddArch(float* address, const float val) {
        atomicAdd(address, val);
    }
#endif

/**
 * @brief Reduce the values of a shared memory array to a single sum stored in
 * the first index.
 *
 * @param shared_data The array shared by the entire thread block containing
 * the elements to be summed.
 * @param block_index The index of the block in which this thread exists.
 */
DEV void warp_reduce(volatile float* shared_data, int block_index) {
    if (BLOCK_SIZE >= 64) shared_data[block_index] += shared_data[block_index + 32];
    if (BLOCK_SIZE >= 32) shared_data[block_index] += shared_data[block_index + 16];
    if (BLOCK_SIZE >= 16) shared_data[block_index] += shared_data[block_index + 8];
    if (BLOCK_SIZE >= 8) shared_data[block_index] += shared_data[block_index + 4];
    if (BLOCK_SIZE >= 4) shared_data[block_index] += shared_data[block_index + 2];
    if (BLOCK_SIZE >= 2) shared_data[block_index] += shared_data[block_index + 1];
}

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
