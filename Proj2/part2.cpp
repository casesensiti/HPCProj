#include <iostream>
#include <ctime>
using namespace std;

void dump (double *c, int n);
void trans(double* dst, double* src, int n);
bool mydegtrf_blocked(int n, double *A, int* IPIV, int step);
void mydtrsm(int n, double *A, double *b, int* piv, bool isForward, double* res);
double verify (double *c0, double *c1, int m, int n);

void dgemm31 (double *a, double *b, double *c, int i0, int j0, int kStart, int kEnd, int n, int B);

int main(int argc, char* argv[])
{
    int n; // width of matrix
    if (argc == 1) n = 1000;
    else {
        n = std::atoi(argv[1]);
    }
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

    // solve using LAPACK
    char    TRANS = 'N';
    int     INFO = 0;
    int     LDA = n;
    int     LDB = n;
    int     N = n;


    LAPACK_dgetrf(&N,&N,A,&LDA,IPIV,&INFO);

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
    printf("Solving using mydegtrf\n");

    // reset IPIV
    for (int i = 0; i < n ; i++) IPIV[i] = i; 

    clock_t start = clock();
    //mydegtrf(3, AT, IPIV);
    mydegtrf_blocked(n, AT, IPIV, 10);
    double elapse = clock() - start;
    printf("Time used in factorization: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
    printf("Performance (in Gflops): %f\n", n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));

    mydtrsm(n, AT, bCopy, IPIV, true, res);
    mydtrsm(n, AT, res, IPIV, false, bCopy);
    printf("Maximum diff between results using LAPCAK and mydegtrf: %.15f\n", verify(b, bCopy, n, 1));

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
// step is the how many lines we process as a block 

// pivot的swap？
bool mydegtrf_blocked(int n, double *A, int* IPIV, int step) {
    for (int i = 0; i < n; i+=step) {
        int ie = i + step < n? i + step : n;
        // pivoting
        for (int i1 = i; i1 < ie; i1++) {
            int maxind = i1;
            double max = A[i1 * n + i1] > 0? A[i1 * n + i1] : -1 * A[i1 * n + i1];
            for (int j = i1 + 1; j < n; j++) {
                double tmp = A[j * n + i1] > 0? A[j * n + i1] : -1 * A[j * n + i1];
                if (tmp > max) {
                    maxind = j;
                    max = tmp;
                }
            }
            if (max < 1e-5 && max > -1e-5) {
                cout << "LUfactorization failed :coefficient matrix is singular or not stable\n";
                return false; // factorization 
            }
            else if (maxind != i1) {
                // save pivoting information
                int tmp = IPIV[i1];
                IPIV[i1] = IPIV[maxind];
                IPIV[maxind] = tmp;
                // swap rows
                double tmpd;
                for (int j = 0; j < n; j++) {
                    tmpd = A[i1 * n + j];
                    A[i1 * n + j] = A[maxind * n + j];
                    A[maxind * n + j] = tmpd;
                }
            }
            // update rows below
            for (int j = i1 + 1; j < n; j++) {
                A[j * n + i1] = A[j * n + i1] / A[i1 * n + i1];
                for (int k = i1 + 1; k < ie; k++) {
                    A[j * n + k] -= A[j * n + i1] * A[i1 * n + k];
                }
            }
        }
        // update rows to the right
        for (int i1 = i; i1 < ie - 1; i1++) {
            for (int j = i1 + 1; j < ie; j++) {
                register double r = A[j * n + i1];
                for (int k = ie; k < n; k++) {
                    A[j * n + k] -= r * A[i1 * n + k];
                }
            }
        }
        // update the lower-right chunk, using pervious algorithm
        dgemm31(A, A, A, ie, ie, i, ie, n, 60);
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


// function dgemm31 - blocking ikj with register reuse
// require B to be multiple of 4
// bounds saves boundary of this computation. 
// i0, j0 and kStart are start pos of i, j and k. kEnd is the end pos of k.
void dgemm31 (double *a, double *b, double *c, int i0, int j0, int kStart, int kEnd, int n, int B) {
    for (int i = i0; i < n; i+=B) {
        for (int k = kStart; k < kEnd; k+=B) { 
            for (int j = j0; j < n; j+=B) {
                // compute boundary of inner loop
                int ie = i + B < n? i + B : n;
                int je = j + B < n? j + B : n;
                int ke = k + B < kEnd? k + B : kEnd;
                // if this block is not multiple of 4 * 2 matrix, use default method
                if (((ie - i) % 4 != 0) || ((je - j) % 2 != 0) || ((ke - k) % 2 != 0)) {
                    for (int i1 = i; i1 < ie; i1++) {
                        for (int k1 = k; k1 < ke; k1++) {
                            register double r = a[i1 * n + k1];
                            for (int j1 = j; j1 < je; j1++)
                                c[i1 * n + j1] -= r * b[k1 * n + j1];
                        }
                    }
                }
                // else, use register reuse 
                else {
                    for (int i1 = i; i1 < ie; i1+=4) {
                        for (int j1 = j; j1 < je; j1+=2) {
                            register double c00 = c[i1 * n + j1];
                            register double c01 = c[i1 * n + j1 + 1];
                            register double c10 = c[(i1 + 1) * n + j1];
                            register double c11 = c[(i1 + 1) * n + j1 + 1];
                            register double c20 = c[(i1 + 2) * n + j1];
                            register double c21 = c[(i1 + 2) * n + j1 + 1];
                            register double c30 = c[(i1 + 3) * n + j1];
                            register double c31 = c[(i1 + 3) * n + j1 + 1];
                            for (int k1 = k; k1 < ke; k1+=2) {
                                register double a0 = a[i1 * n + k1];
                                register double a1 = a[(i1 + 1) * n + k1];
                                register double a2 = a[(i1 + 2) * n + k1];
                                register double a3 = a[(i1 + 3) * n + k1];

                                register double b0 = b[k1 * n + j1];
                                register double b1 = b[k1 * n + j1 + 1];

                                //printf("i: %d, j: %d, k: %d, a: %f %f %f %f, b: %f %f\n", i, j, k, a0, a1, a2, a3, b0, b1);

                                c00 -= a0 * b0;
                                c01 -= a0 * b1;
                                c10 -= a1 * b0;
                                c11 -= a1 * b1;
                                c20 -= a2 * b0;
                                c21 -= a2 * b1;
                                c30 -= a3 * b0;
                                c31 -= a3 * b1;
                                // get second group of elements
                                a0 = a[i1 * n + k1 + 1];
                                a1 = a[(i1 + 1) * n + k1 + 1];
                                a2 = a[(i1 + 2) * n + k1 + 1];
                                a3 = a[(i1 + 3) * n + k1 + 1];

                                b0 = b[(k1 + 1) * n + j1];
                                b1 = b[(k1 + 1) * n + j1 + 1];

                                //printf("i: %d, j: %d, k: %d, a: %f %f %f %f, b: %f %f\n", i, j, k, a0, a1, a2, a3, b0, b1);

                                c00 -= a0 * b0;
                                c01 -= a0 * b1;
                                c10 -= a1 * b0;
                                c11 -= a1 * b1;
                                c20 -= a2 * b0;
                                c21 -= a2 * b1;
                                c30 -= a3 * b0;
                                c31 -= a3 * b1;
                            }
                            c[i1 * n + j1] = c00;
                            c[i1 * n + j1 + 1] = c01;
                            c[(i1 + 1) * n + j1] = c10;
                            c[(i1 + 1) * n + j1 + 1] = c11;
                            c[(i1 + 2) * n + j1] = c20;
                            c[(i1 + 2) * n + j1 + 1] = c21;
                            c[(i1 + 3) * n + j1] = c30;
                            c[(i1 + 3) * n + j1 + 1] = c31;
                        }
                    }
                }
            }
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
void dump (double *c, int n) {
    for (int i = 0; i < n; i++) {
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


