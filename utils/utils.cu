/** Several utility functions.
 *
 * utils.cu:
 *  - Defines several host- and device-side (CUDA) utility functions.
 */

#include "utils.cuh"


//---------------------------------------------------------------------------//
// Device-side CUDA utility functions.                                       //
//---------------------------------------------------------------------------//

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


//---------------------------------------------------------------------------//
// Host-side CUDA utility functions.                                         //
//---------------------------------------------------------------------------//

/**
 * @brief Get the number of GPU devices available on this machine.
 *
 * @param num_devices Integer used to store the number of GPU devices.
 */
HST void get_number_devices(int *num_devices) {
    cudaError_t err = cudaGetDeviceCount(num_devices);
    if (err != cudaSuccess) {
        *num_devices = 0;
    }
}

/**
 * @brief Get the compute capability of a specific GPU device.
 *
 * @param device The ID of the GPU device.
 * @return int The compute capability of the GPU device (e.g. 5.2 = 52).
 */
HST int get_compute_capability(const int device) {
    cudaDeviceProp dprop;
    CUDA_CHECK(cudaGetDeviceProperties(&dprop, device));

    return std::stoi(std::to_string(dprop.major) + std::to_string(dprop.minor));
}

/**
 * @brief Get the size of a warp on a specific GPU device.
 *
 * @param device The ID of the GPU device.
 * @return int The size of a warp (e.g. 32).
 */
HST int get_warp_size(const int device) {
    cudaDeviceProp dprop;
    CUDA_CHECK(cudaGetDeviceProperties(&dprop, device));

    return dprop.warpSize;
}

/**
 * @brief Get the memory usage of a specific GPU device.
 *
 * @param device_id The ID of the GPU device.
 * @param free_memory The free memory of the GPU device.
 * @param total_memory The total memory of the GPU device.
 * @param rounding The number by which the free and total memory will be rounded.
 */
HST void get_device_memory(const int& device_id, size_t& free_memory, size_t& total_memory, const size_t rounding) {
    int old_device;
    CUDA_CHECK(cudaGetDevice(&old_device));
    CUDA_CHECK(cudaSetDevice(device_id));

    size_t free_bytes, total_bytes;

    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    free_memory = free_bytes / rounding;
    total_memory = total_bytes / rounding;

    CUDA_CHECK(cudaSetDevice(old_device));
}


//---------------------------------------------------------------------------//
// Host-side utility functions.                                              //
//---------------------------------------------------------------------------//

/**
 * @brief Get the memory usage of the host machine.
 *
 * @param free_memory The free memory of the host.
 * @param total_memory The total memory of the host.
 * @param rounding The number by which the free and total memory will be rounded.
 */
HST void get_host_memory(size_t& free_memory, size_t& total_memory, const size_t rounding) {
    size_t free_bytes, total_bytes;

    total_bytes = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    free_bytes = sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE);

    total_memory = total_bytes / rounding;
    free_memory = free_bytes / rounding;
}

/**
 * @brief Print a help message.
 */
HST void show_help_msg(void) {
    std::cout << "Usage: gpucmt [options]...\n" <<
    "Train EM-based click models using multiple GPUs and/or machines.\n" <<
    "Example: ./gpucmt -r 'dataset.txt' -s 40000 -i 50 -m 0 -p 0 -t 0.2 -n 32 -j 0\n\n" <<
    "Options:\n" <<
    "  -h, --help\t\t\tDisplay this help message.\n" <<
    "  -r, --raw-path\t\tPath to the dataset to use.\n" <<
    "  -o, --output-path\t\tPath to the output file for the trained\n" <<
    "\t\t\t\tparameters.\n" <<
    "  -s, --max-sessions\t\tMaximum number of query sessions to read from\n" <<
    "\t\t\t\tthe dataset.\n" <<
    "  -n, --n-threads\t\tNumber of threads per machine.\n" <<
    "  -i, --itr\t\t\tNumber of iterations to run.\n" <<
    "  -m, --model-type\t\tClick model type to use.\n" <<
    "\t\t\t\t0: PBM, 1: CCM, 2: DBN, 3: UBM.\n" <<
    "  -p, --partition-type\t\tDataset partitioning scheme to use.\n" <<
    "\t\t\t\t0: Round-Robin, 1: Maximum Utilization,\n" <<
    "\t\t\t\t2: Resource-Aware Maximum Utilization,\n" <<
    "\t\t\t\t3: Newest architecture first.\n" <<
    "  -t, --test-share\t\tShare of the dataset to use for testing.\n" <<
    "  -j, --job-id\t\t\tJob ID to use for logging.\n" <<
    "  -e, --exec-mode\t\tExecution mode.\n" <<
    "\t\t\t\t0: GPU, 1: CPU, 2: Hybrid.\n" << std::endl;
    // "  -v, --verbose\t\t\tVerbose mode.\n" << std::endl;
}