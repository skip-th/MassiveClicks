/** Several utility functions.
 *
 * utils.cu:
 *  - Defines several host- and device-side (CUDA) utility functions.
 */

#include "utils.cuh"


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
 * @return The compute capability of the GPU device (e.g. 5.2 = 52).
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
 * @return The size of a warp (e.g. 32).
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
 * @param rounding The number by which the free and total memory will be
 * rounded.
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
    const std::string help_message = R"(Usage: mclicks [options]...
Train EM-based click models using multiple GPUs and/or machines.
Example: ./mclicks -r 'dataset.txt' -s 40000 -i 50 -m 0 -p 0 -t 0.2 -n 32 -j 0

Options:
  -h, --help               Display this help message.
  -r, --raw-path           Path to the dataset to use.
  -o, --output-path        Path to the output file for the trained parameters.
  -s, --max-sessions       Maximum number of query sessions to read from the dataset.
  -g, --n-gpus             Maximum number of GPUs per machine.
  -n, --n-threads          Number of threads per machine.
  -i, --itr                Number of iterations to run.
  -m, --model-type         Click model type to use. 0: PBM, 1: CCM, 2: DBN, 3: UBM.
  -p, --partition-type     Dataset partitioning scheme to use.
                           0: Round-Robin, 1: Maximum Utilization,
                           2: Resource-Aware Maximum Utilization,
                           3: Newest architecture first.
  -t, --test-share         Share of the dataset to use for testing.
  -j, --job-id             Job ID to use for logging.
  -e, --exec-mode          Execution mode. 0: GPU, 1: CPU, 2: Hybrid.
)";
//   -v, --verbose            Verbose mode.";

    std::cout << help_message << std::endl;
}