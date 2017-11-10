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
    int p, id, rc, j, global_count = 0;
    int B = 400000;
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
    long long low = (id * nElements) / p; // left most global index for this processor
    long long high = ((id + 1) * nElements) / p - 1;
    long long size = high - low + 1;
    int sieveN = sqrt(n);
    int sieveLen = (sieveN + 1) / 2 - 1;
    
    // prepare space 
    char *sieves = malloc(sizeof(char) * sieveLen); // this sieves array is used for traditional algorithm 
    if (!sieves) {
        printf("Cannot allocate enough memory.\n");
        MPI_Finalize();
        exit(1);
    }
    memset(sieves, 0, sieveLen);
    char *mark = (char*) malloc(sizeof(char) * size);
    if (!mark) {
        printf("Cannot allocate enough memory.\n");
        MPI_Finalize();
        exit(1);
    }
    memset(mark, 0, size);
    // find sieves locally
    findSieves(sieves, sieveLen);
    long long numSieves = 0;
    for (i = 0; i < sieveLen; i++) if(!sieves[i]) numSieves++; 
    int *realSieves = malloc(sizeof(int) * numSieves); // this array saves every prime consecutively
    if (!realSieves) {
        printf("Cannot allocate enough memory.\n");
        MPI_Finalize();
        exit(1);
    }
    int tmpIdx = 0;
    for (j = 0; j < sieveLen; j++) if (!sieves[j]) realSieves[tmpIdx++] = 2 * j + 3;
    // main loop
    int sieveL, sieveR;
    for (sieveL = 0; sieveL < numSieves; sieveL += 10000) {
        sieveR = sieveL + 10000 - 1;
        if (sieveR >= numSieves) sieveR = numSieves - 1;
        long long blockL, blockR;
        for (blockL = 0; blockL < size; blockL += B) {
            blockR = blockL + B - 1;
            if (blockR >= size) blockR = size - 1;
            for (j = sieveL; j <= sieveR; j++) {
                long long prime = realSieves[j];
                long long index = (prime * prime - 3) / 2; 
                long long first;
                if (index >= low + blockL) first = index - low - blockL; 
                else {
                    first = prime - (low + blockL - index) % prime;
                    first %= prime;
                }
                for (i = first + blockL; i <= blockR; i += prime) mark[i] = 1;
            }
        }
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
