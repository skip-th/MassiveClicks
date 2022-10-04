/** Several utility functions.
 *
 * utility_functions.cpp:
 *  - Defines several utility functions.
 */


#include "utility_functions.h"

/**
 * @brief Print a help message.
 */
void print_help_msg(void) {
    std::cout << "Usage: gpucmt [options]...\n" <<
    "Train EM-based click models using multiple GPUs and/or machines.\n" <<
    "Example: ./gpucmt -r 'dataset.txt' -s 100000 -i 50 -m 1 -p 2\n\n" <<
    "Options:\n" <<
    "  -h, --help\t\t\tDisplay this help message.\n" <<
    "  -r, --raw-path\t\tPath to the dataset to use.\n" <<
    "  -s, --max-sessions\t\tMaximum number of query sessions to read from\n" <<
    "\t\t\t\tthe dataset.\n" <<
    "  -i, --itr\t\t\tNumber of iterations to run.\n" <<
    "  -m, --model-type\t\tClick model type to use.\n" <<
    "\t\t\t\t0: PBM, 1: CCM, 2: DBN, 3: UBM.\n" <<
    "  -p, --partition-type\t\tDataset partitioning scheme to use.\n" <<
    "\t\t\t\t0: Round-Robin, 1: Maximum Utilization,\n" <<
    "\t\t\t\t2: Resource-Aware Maximum Utilization.\n" <<
    "  -t, --test-share\t\tShare of the dataset to use for testing.\n" <<
    "  -j, --job-id\t\t\tJob ID to use for logging.\n" << std::endl;
    // "  -v, --verbose\t\t\tVerbose mode.\n" << std::endl;
}