/** CUDA and MPI macro's.
 *
 * macros.cpp:
 *  - Defines the MPI abort functions, used to cleanly abort an MPI process.
 */

#include <mpi.h>

#include "macros.cuh"

/**
 * @brief Shut down MPI cleanly if something goes wrong.
 *
 * @param err Error code.
 */
void mpi_abort(const int err) {
    std::cout << "Quiting MPI\n";

    MPI_Abort(MPI_COMM_WORLD, err);
}