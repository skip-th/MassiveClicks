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
        #define ROOT 0
        #define BLOCK_SIZE 96 // Can't be less than MAX_SERP_LENGTH.

        #define D2H 0
        #define H2D 1

        #define PUBLIC 0
        #define PRIVATE 1
        #define ALL -1

        #define PARAM_DEF_NUM 1
        #define PARAM_DEF_DENOM 2

        #define MAX_SERP_LENGTH 10
        #define QUERY_LINE_LENGTH 15
        #define CLICK_LINE_LENGTH 4

        #include <vector>
        template<typename T> using NetworkMap = std::vector<std::vector<T>>; // Node ID -> Device ID -> Pointer.
#endif // CLICK_MODEL_DEFINITIONS_H