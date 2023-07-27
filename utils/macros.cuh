/** CUDA and MPI macro's.
 *
 * macros.cuh:
 *  - Defines the CUDA and MPI error handling macro's.
 */

#ifndef CUDA_MPI_MACROS_H
#define CUDA_MPI_MACROS_H

// System include.
#include <iostream>
#include <cuda_runtime.h>
#include <mpi.h>

// User include.
#include "../utils/definitions.h"

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

#endif // CUDA_MPI_MACROS_H