#include "mpi.h"
#include <stdio.h>
#include <math.h>

void findSieves(char* sieves, int len) {
    int prime = 3;
    int index = 0;
    while (index < len && prime * prime <= 2 * len + 1) {
        for (int i = (prime * prime - 3) / 2; i < len; i += prime) sieves[i] = 1;  
        while (sieves[++index]);
        prime = 2 * index + 3;
    }
}
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
    int sieveN = sqrt(n);
    int sieveLen = (sieveN + 1) / 2 - 1;
    
    // prepare space 
    char *sieves = malloc(sizeof(char) * sieveLen);
    if (!sieves) {
        printf("Cannot allocate enough memory.\n");
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i < sieveLen; i++) sieves[i] = 0;    

    char *mark = (char*) malloc(sizeof(char) * size);
    if (!mark) {
        printf("Cannot allocate enough memory.\n");
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i < size; i++) mark[i] = 0;    
    // find sieves locally
    findSieves(sieves, sieveLen);
    // main loop
    long long prime = 3;
    long long index = 0;

    while (index < sieveLen) {
        long long start = (prime * prime - 3) / 2;
        long long first = 0;
        if (start < low) {
            first = prime - (low - start) % prime; 
            first %= prime;
        } else first = start - low;
        for (i = first; i < size; i+= prime) {
            mark[i] = 1; 
        }
        while (sieves[++index]); 
        prime = 2 * index + 3;
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
