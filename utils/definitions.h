/** Program global definitions.
 *
 * definitions.h:
 *  - Defines the defintions used throughout the program.
 *  - CUDA specific macros.
 *  - Contants for program logic.
 */

#ifndef CLICK_MODEL_DEFINITIONS_H
#define CLICK_MODEL_DEFINITIONS_H

#ifdef __CUDACC__ // If compiling with nvcc.
    #define HST __host__
    #define DEV __device__
    #define GLB __global__
    #define SHR __shared__
    #define CST __constant__
#else
    #define HST
    #define DEV
    #define GLB
    #define SHR
    #define CST
#endif // __CUDACC__

// Constants for CUDA.
#define BLOCK_SIZE 128 // Threads per block. Can't be less than MAX_SERP.

// Constants for MPI.
#define ROOT_RANK 0 // The id of the root node.

// Constants for data transfer directions.
#define D2H 0 // Device to host.
#define H2D 1 // Host to device.

// Constants for parameter types.
#define PUBLIC 0
#define PRIVATE 1
#define ALL -1

// Constants for parameter definition.
#define PARAM_DEF_NUM 1.f // Default numerator for parameters.
#define PARAM_DEF_DENOM 2.f // Default denominator for parameters.

// Constants for click model specific values.
#define N_TAU 3 // Number of continuation (tau) parameters used in CCM.
#define N_GAM 1 // Number of continuation (gamma) parameters used in DBN.

// Constants defining the dataset layout.
#define MAX_SERP 10 // Maximum number of documents in a SERP.
#define QUERY_LINE_LENGTH 15 // Maximum number of characters in a raw text query line.
#define CLICK_LINE_LENGTH 4 // Maximum number of characters in a raw text click line.

#endif // CLICK_MODEL_DEFINITIONS_H
