#include "mpi.h"
#include <stdio.h>

int main (int argc, char* argv[]) {
    int p, id, rc;
    // mpi init
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    printf("Processes: %d, id: %d\n", p, id);
    //fflush();

    MPI_Finalize();
    return 0;
}