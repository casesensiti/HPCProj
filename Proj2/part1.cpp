#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include "lapacke.h"
#include "blas.h"
using namespace std;

void dump (double *c, int m, int n);
void trans(double* dst, double* src, int n);
bool mydegtrf(int n, double *A, int* IPIV);
void mydtrsm(int n, double *A, double *b, int* piv, bool isForward, double* res);
double verify (double *c0, double *c1, int m, int n);

int main(int argc, char* argv[])
{
    int n; // width of matrix
    if (argc == 1) n = 1000;
    else {
        n = std::atoi(argv[1]);
    }
    printf("Matrix size: %d * %d\n", n, n);
    // allocate memory
    double *A = (double*) malloc(sizeof(double) * n * n);
    double *AT = (double*) malloc(sizeof(double) * n * n);
    double *b = (double*) malloc(sizeof(double) * n);
    double *bCopy = (double*) malloc(sizeof(double) * n);
    int *IPIV = (int*) malloc(sizeof(int) * n);
    double *res = (double*) malloc(sizeof(double) * n);
    // initialize matrics and vectors for solving Ax = b
    for (int i = 0; i < n * n; i++) A[i] = (rand() % 100) / 100.0; 
    for (int i = 0; i < n ; i++) {
        b[i] = (rand() % 100) / 100.0; 
        bCopy[i] = b[i];
    }
    trans(AT, A, n); // transpose A into AT, A is used for LAPACK, AT is used for my implementation
    clock_t start;
 
    // solve using LAPACK
    char    TRANS = 'N';
    int     INFO = 0;
    int     LDA = n;
    int     LDB = n;
    int     N = n;


    printf("Solving using lapack degtrf\n");
    start = clock();
    LAPACK_dgetrf(&N,&N,A,&LDA,IPIV,&INFO);
    //mydegtrf_blocked(n, AT, IPIV, 10);
    double elapse = clock() - start;
    printf("Time used in factorization: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
    printf("Performance (in Gflops): %f\n", (2.0/3) * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));


    char     SIDE = 'L';
    char     UPLO = 'L';
    char     DIAG = 'U';
    int      M    = 1;
    double   a    = 1.0;

    for(int i = 0; i < N; i++)
    {
        double tmp = b[IPIV[i]-1];
        b[IPIV[i]-1] = b[i];
        b[i] = tmp;
    }

    // forward  L(Ux) = B => y = Ux
    dtrsm_(&SIDE,&UPLO,&TRANS,&DIAG,&N,&M,&a,A, &N, b, &N);
    UPLO = 'U';
    DIAG = 'N';
    // backward Ux = y
    dtrsm_(&SIDE,&UPLO,&TRANS,&DIAG,&N,&M,&a,A, &N, b, &N);


    // solve using mydegtrf
    printf("\nSolving using mydegtrf\n");

    // reset IPIV
    for (int i = 0; i < n ; i++) IPIV[i] = i; 

    start = clock();
    mydegtrf(n, AT, IPIV);
    //mydegtrf_blocked(n, AT, IPIV, 10);
    elapse = clock() - start;
    printf("Time used in factorization: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
    printf("Performance (in Gflops): %f\n", (2.0/3) * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));

    mydtrsm(n, AT, bCopy, IPIV, true, res);
    mydtrsm(n, AT, res, IPIV, false, bCopy);
    printf("Maximum diff between results using LAPCAK and mydegtrf: %.15f\n\n", verify(b, bCopy, n, 1));

    // free memory
    free(A);
    free(AT);
    free(b);
    free(bCopy);
    free(IPIV);
    free(res);
    return 0;
}

// n is the width of the matrix, A is the matrix to be factorized
// elements are arranged in row-order
// IPIV is used to record pivot information. At the beginning in IPIV, elements should be 1:n-1
bool mydegtrf(int n, double *A, int* IPIV) {
    for (int i = 0; i < n - 1; i++) {
        // pivoting
        int maxind = i;
        double max = A[i * n + i] > 0? A[i * n + i] : -1 * A[i * n + i];
        for (int j = i + 1; j < n; j++) {
            double tmp = A[j * n + i] > 0? A[j * n + i] : -1 * A[j * n + i];
            if (tmp > max) {
                maxind = j;
                max = tmp;
            }
        }
        if (max < 1e-5 && max > -1e-5) {
            cout << "LUfactorization failed :coefficient matrix is singular or not stable\n";
            return false; // factorization 
        }
        else if (maxind != i) {
            // save pivoting information
            int tmp = IPIV[i];
            IPIV[i] = IPIV[maxind];
            IPIV[maxind] = tmp;
            // swap rows
            double tmpd;
            for (int j = 0; j < n; j++) {
                tmpd = A[i * n + j];
                A[i * n + j] = A[maxind * n + j];
                A[maxind * n + j] = tmpd;
            }
        }

        // factorization
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] = A[j * n + i] / A[i * n + i];
            for (int k = i + 1; k < n; k++) {
                A[j * n + k] -= A[j * n + i] * A[i * n + k];
            }
        }
    }
    return true;
}

// to resolve Ax = b, n is the length of the matrix, res is x, piv is the pivot record
void mydtrsm(int n, double *A, double *b, int* piv, bool isForward, double* res) {
    if (isForward) {
        res[0] = b[piv[0]];
        for (int i = 1; i < n; i++) {
            double tmpb = b[piv[i]];
            for (int j = 0; j < i; j++) {
                tmpb -= res[j] * A[i * n + j];
            }
            res[i] = tmpb;
        }
    } else {
        res[n - 1] = b[n - 1] / A[n * n - 1];
        for (int i = n - 2; i >= 0; i--) {
            double tmpb = b[i];
            for (int j = n - 1; j > i; j--) {
                tmpb -= res[j] * A[i * n + j];
            }
            res[i] = tmpb / A[i * n + i];
        }
    }
}

// interchange between column-order and row-order
void trans(double* dst, double* src, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i * n + j] = src[j * n + i];
        }
    }
}

// print the matrix, helps debug
void dump (double *c, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%f ", c[i * n + j]);
        printf("\n");
    }
    printf("\n");
}

// get the maximum diff
double verify (double *c0, double *c1, int m, int n) {
    double diff = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double tmp = c0[i * n + j] - c1[i * n + j];
            tmp = tmp > 0? tmp : -1 * tmp;
            if (tmp > diff) diff = tmp;
        }
    }
    return diff;
}


