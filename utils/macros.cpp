/** CUDA and MPI macro's.
 *
 * macros.cpp:
 *  - Defines the MPI abort functions, used to cleanly abort an MPI process.
 */

#include <mpi.h>
#include "macros.cuh"

/**
 * @brief Terminate all MPI processes in case of an irrecoverable error.
 *
 * @param err Error code.
 */
void mpi_abort(const int err) {
    std::cout << "Quiting MPI" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, err);
}