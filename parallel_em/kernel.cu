/** CUDA kernel functions.
 *
 * kernel.cu:
 *  - Defines several CUDA kernels usable when training a click model.
 */

#include "kernel.cuh"

//---------------------------------------------------------------------------//
// Device-side computation.                                                  //
//---------------------------------------------------------------------------//

namespace Kernel {
    /**
     * @brief A CUDA kernel which initializes the click model on the GPU and stores
     * the references to allocated memory relevant to calculating the click model.
     *
     * @param model_type The type of click model (e.g. 0 = PBM).
     * @param node_id The MPI communication rank of this node.
     * @param device_id The ID of the GPU device.
     * @param cm_param_ptr The pointers to memory allocated for usage by the click model.
     * @param parameter_sizes The size of the allocated memory for the click model.
     */
    GLB void initialize(const int model_type, const int node_id, const int device_id, Param** cm_param_ptr, int* parameter_sizes) {
        // Initialize click model.
        cm_dev = create_cm_dev(model_type);
        // Set the device parameter pointers for the click model.
        cm_dev->set_parameters(cm_param_ptr, parameter_sizes);
        // Print a confirmation message on the first device of the root node.
        if (node_id == ROOT && device_id == 0) {
            cm_dev->say_hello();
        }
    }

    /**
     * @brief A CUDA kernel which performs the Expectation-Maximization training
     * steps for the chosen click model.
     *
     * @param partition A pointer to the partition containing the dataset to train the click model on.
     * @param partition_size The size of the partition containin the dataset.
     */
    GLB void em_training(SearchResult_Dev* partition, int partition_size) {
        // Calculate the starting index within the query session array for this thread.
        int thread_index = blockDim.x * blockIdx.x + threadIdx.x; // Global index.

        // End this thread if it can't be assigned a query session.
        if (thread_index >= partition_size)
            return;

        // Retrieve the search results corresponding to the current query from
        // the dataset.
        SERP_Dev query_session = SERP_Dev(partition, partition_size, thread_index);

        // Estimate click model parameters for the given query session.
        cm_dev->process_session(query_session, thread_index, partition_size);
    }

    /**
     * @brief A CUDA kernel which updates the global parameter values using the
     * parameters estimated locally on each thread.
     *
     * @param partition A pointer to the partition containing the dataset to train the click model on.
     * @param partition_size The size of the partition containin the dataset.
     * @param parameter_type The type of parameter being updated.
     */
    GLB void update(SearchResult_Dev* partition, int partition_size) {
        // Calculate the starting index within the query session array for this thread.
        int thread_index = blockDim.x * blockIdx.x + threadIdx.x; // Global index.
        int block_index = threadIdx.x; // Local index (local to thread block).

        // Retrieve the search results corresponding to the current query from
        // the dataset.
        SERP_Dev query_session = SERP_Dev(partition, partition_size, thread_index);

        // Estimate click model examination parameters.
        cm_dev->update_parameters(query_session, thread_index, block_index, partition_size);
    }
}