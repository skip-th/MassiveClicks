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
     * @param model_type The type of click model (e.g., 0 = PBM).
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
        if (node_id == ROOT_RANK && device_id == 0) {
            cm_dev->say_hello();
        }
    }

    /**
     * @brief A CUDA kernel which performs the Expectation-Maximization training
     * steps for the chosen click model.
     *
     * @param dataset A pointer to the dataset used to train the click model on.
     * @param dataset_size The size of the dataset containin the dataset.
     */
    GLB void em_training(SearchResult_Dev* dataset, int dataset_size) {
        // Calculate the starting index within the query session array for this thread.
        int thread_index = blockDim.x * blockIdx.x + threadIdx.x; // Global index.

        // End this thread if it can't be assigned a query session.
        if (thread_index >= dataset_size)
            return;

        // Retrieve the search results corresponding to the current query from
        // the dataset.
        SERP_Dev query_session = SERP_Dev(dataset, dataset_size, thread_index);
        SHR char clicks[BLOCK_SIZE * MAX_SERP]; // Click per rank.
        SHR int pidx[BLOCK_SIZE * MAX_SERP]; // Parameter index.

        #pragma unroll
        for (int rank = 0; rank < MAX_SERP; rank++) {
            clicks[rank * BLOCK_SIZE + threadIdx.x] = query_session[rank].get_click();
            pidx[rank * BLOCK_SIZE + threadIdx.x] = query_session[rank].get_param_index();
        }

        // Estimate click model parameters for the given query session.
        cm_dev->process_session(query_session, thread_index, dataset_size, clicks, pidx);
    }

    /**
     * @brief A CUDA kernel which updates the global parameter values using the
     * parameters estimated locally on each thread.
     *
     * @param dataset A pointer to the dataset used to train the click model on.
     * @param dataset_size The size of the dataset containin the dataset.
     */
    GLB void update(SearchResult_Dev* dataset, int dataset_size) {
        // Calculate the starting index within the query session array for this thread.
        int thread_index = blockDim.x * blockIdx.x + threadIdx.x; // Global index.
        int block_index = threadIdx.x; // Local index (local to thread block).

        // Retrieve the parameter indices shared between similar qd-pairs from
        // the dataset.
        SHR int pidx[BLOCK_SIZE * MAX_SERP];

        #pragma unroll
        for (int rank = 0; rank < MAX_SERP; rank++) {
            pidx[rank * BLOCK_SIZE + threadIdx.x] = dataset[rank * dataset_size + thread_index].get_param_index();
        }

        // Estimate click model examination parameters.
        cm_dev->update_parameters(thread_index, block_index, dataset_size, pidx);
    }
}