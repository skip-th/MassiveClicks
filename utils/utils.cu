/** Several utility functions.
 *
 * utils.cu:
 *  (Defines several host- and device-side (CUDA) utility functions.)
 */

#include "utils.cuh"

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

/**
 * @brief Get the number of CUDA cores per SM based on the compute capability.
 *
 * @param cc_major The major compute capability.
 * @return int The number of CUDA cores per SM.
 */
HST int get_cores_per_SM(int cc_major) {
    switch (cc_major) {
        case 1: // Tesla
            return 8;
        case 2: // Fermi
            return 32;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            return 64;
        case 7: // Volta and Turing
            return 64;
        case 8: // Ampere
            return 128;
        default: // Unknown architecture
            return 128; // Assume newer than Ampere
    }
}

/**
 * @brief Get the device properties object.
 *
 * @param device_id The ID of the GPU device.
 * @return DeviceProperties The properties of the GPU device.
 */
HST DeviceProperties get_device_properties(const int device_id) {
    int old_device;
    CUDA_CHECK(cudaGetDevice(&old_device));
    CUDA_CHECK(cudaSetDevice(device_id));
    DeviceProperties properties;
    cudaDeviceProp dprop;
    CUDA_CHECK(cudaGetDeviceProperties(&dprop, device_id));
    CUDA_CHECK(cudaMemGetInfo(&properties.available_memory, &properties.total_global_memory)); // bytes
    properties.device_id = device_id;
    properties.compute_capability = dprop.major * 10 + dprop.minor; // e.g., 52 = 5.2
    properties.shared_memory_per_block = dprop.sharedMemPerBlock; // bytes
    properties.total_constant_memory = dprop.totalConstMem; // bytes
    properties.registers_per_block = dprop.regsPerBlock;
    properties.registers_per_sm = dprop.regsPerMultiprocessor;
    properties.threads_per_block = dprop.maxThreadsPerBlock;
    properties.threads_per_sm = dprop.maxThreadsPerMultiProcessor;
    properties.warp_size = dprop.warpSize;
    properties.memory_clock_rate = (size_t) dprop.memoryClockRate * 1000; // Hz
    properties.memory_bus_width = dprop.memoryBusWidth; // bits
    properties.cores_per_sm = get_cores_per_SM(dprop.major);
    properties.clock_rate = dprop.clockRate * 1000; // Hz
    properties.multiprocessor_count = dprop.multiProcessorCount;
    properties.peak_performance = ((size_t) properties.cores_per_sm * (size_t) properties.multiprocessor_count) // CUDA cores
                                  * ((size_t) properties.clock_rate) // Clock rate (Hz)
                                  * 2.f; // FMA
    strcpy(properties.device_name, dprop.name);
    CUDA_CHECK(cudaSetDevice(old_device));
    return properties;
}

/**
 * @brief Get the host properties object.
 *
 * @param node_id The ID of the node.
 * @return HostProperties The properties of the host.
 */
HST HostProperties get_host_properties(const int node_id) {
    HostProperties properties;
    properties.node_id = node_id;
    properties.thread_count = std::thread::hardware_concurrency();
    properties.free_memory = (sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE));
    properties.total_memory = (sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE));
    gethostname(properties.host_name, HOST_NAME_MAX);
    return properties;
}

/**
 * @brief Get the node properties object.
 *
 * @param node_id The MPI rank of the node.
 * @return NodeProperties The properties of the current node.
 */
HST NodeProperties get_node_properties(const int node_id) {
    NodeProperties properties;
    properties.host = get_host_properties(node_id);

    if (cudaGetDeviceCount(&properties.host.device_count) != cudaSuccess) {
        properties.host.device_count = 0;
    }

    for (int device_id = 0; device_id < properties.host.device_count; device_id++) {
        properties.devices.push_back(get_device_properties(device_id));
    }
    return properties;
}

/**
 * @brief Get the memory usage of the host machine.
 *
 * @param free_memory The free memory of the host.
 * @param total_memory The total memory of the host.
 * @param rounding The number by which the free and total memory will be rounded.
 */
HST void get_host_memory(size_t& free_memory, size_t& total_memory, const size_t rounding) {
    total_memory = (sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)) / rounding;
    free_memory = (sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE)) / rounding;
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
                           2: Proportional Maximum Utilization,
                           3: Newest architecture first.
  -t, --test-share         Share of the dataset to use for testing.
  -j, --job-id             Job ID to use for logging.
  -e, --exec-mode          Execution mode. 0: GPU, 1: CPU, 2: Hybrid.
)";
//   -v, --verbose            Produce more output for diagnostic purposes.";

    std::cout << help_message << std::endl;
}