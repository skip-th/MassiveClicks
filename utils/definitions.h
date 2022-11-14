/** Program global definitions.
 *
 * definitions.h:
 *  - Defines the defintions used throughout the program.
 */


#ifdef __CUDACC__
    #define HST __host__
    #define DEV __device__
    #define GLB __global__
    #define SHR __shared__
#else
    #define HST
    #define DEV
    #define GLB
    #define SHR
#endif // __CUDACC__

#ifndef CLICK_MODEL_DEFINITIONS_H
    #define CLICK_MODEL_DEFINITIONS_H
        #define ROOT 0 // The id of the root node.
        #define BLOCK_SIZE 96 // Threads per block. Can't be less than MAX_SERP.

        #define D2H 0 // Device to host.
        #define H2D 1 // Host to device.

        #define PUBLIC 0
        #define PRIVATE 1
        #define ALL -1

        #define PARAM_DEF_NUM 1 // Default numerator for parameters.
        #define PARAM_DEF_DENOM 2 // Default denominator for parameters.

        #define N_TAU 3 // Number of continuation (tau) parameters used in CCM.
        #define N_GAM 1 // Number of continuation (gamma) parameters used in DBN.

        #define MAX_SERP 10 // Maximum number of documents in a SERP.
        #define QUERY_LINE_LENGTH 15 // Maximum number of characters in a raw text query line.
        #define CLICK_LINE_LENGTH 4 // Maximum number of characters in a raw text click line.

        #include <vector>
        template<typename T> using NetworkMap = std::vector<std::vector<T>>; // Node ID -> Device ID -> Pointer.
#endif // CLICK_MODEL_DEFINITIONS_H