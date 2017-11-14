#include "mpi.h"
#include <stdio.h>
#include <math.h>

int main (int argc, char* argv[]) {
    int p, id, rc, global_count = 0;
    long long i;
    // mpi init
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = -1 * MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // partitioning problem
    long long n = 10000000000;
    long long nElements = (n + 1) / 2 - 1; // suitable for even and odd numbers
    long long low = (id * nElements) / p;
    long long high = ((id + 1) * nElements) / p - 1;
    long long size = high - low + 1;
    // whether all sieving numbers are in process 0
    if (!id) {
        if (2 * size + 1 < sqrt(n)) {
            printf("Too many processes\n");
            MPI_Finalize();
            exit(1);
        }
    }
    // prepare space 
    char *mark = (char*) malloc(sizeof(char) * size);
    if (!mark) {
        printf("Cannot allocate enough memory.\n");
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i < size; i++) mark[i] = 0;    
    // start loop
    long long prime = 3;
    long long index = 0;

    while (index < size && prime * prime <= n) {
        long long start = (prime * prime - 3) / 2;
        long long first = 0;
        if (start < low) {
            first = prime - (low - start) % prime; 
            first %= prime;
        } else first = start - low;
        for (i = first; i < size; i+= prime) {
            mark[i] = 1; 
        }
        if (!id) {
            while (index < size && mark[++index]); 
            prime = 2 * index + 3;
        }
        MPI_Bcast(&prime, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    int count = 0;
    for (i = 0; i < size; i++) if (!mark[i]) count++;
    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 
    elapsed_time += MPI_Wtime();
    
    if (!id) {
        printf("The total number of prime: %d, total time: %f, total node: %d\n", 
            global_count + 1, elapsed_time, p);
    }

    MPI_Finalize();
    return 0;
}
