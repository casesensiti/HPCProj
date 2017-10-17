#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"


void dgemm0 (double *a, double *b, double *c, int n);
void dgemm11 (double *a, double *b, double *c, int n);
void dgemm12 (double *a, double *b, double *c, int n);
void dgemm13 (double *a, double *b, double *c, int n);
void dgemm14 (double *a, double *b, double *c, int n);
void dgemm15 (double *a, double *b, double *c, int n);
void dgemm16 (double *a, double *b, double *c, int n);
void dgemm21 (double *a, double *b, double *c, int n, int B);
void dgemm22 (double *a, double *b, double *c, int n, int B);
void dgemm23 (double *a, double *b, double *c, int n, int B);
void dgemm24 (double *a, double *b, double *c, int n, int B);
void dgemm25 (double *a, double *b, double *c, int n, int B);
void dgemm26 (double *a, double *b, double *c, int n, int B);
void dgemm3 (double *a, double *b, double *c, int n);
double verify (double *, double *, int n);
void dump (double *c, int n);

int main () {
	srand(clock());
	int sizes[] = {2048};
	int blockSizes[] = {10, 20, 40, 80, 160};
	double *a, *b, *c0, *c1, *c2, *c3; // c0 for result of degmm0, c1 for result of degmm1
	for (int i = 0; i < 1; i++) {
		// get size of this iteration
		int n = sizes[i]; // length or width
		printf("Size of matrix: %d * %d\n\n", n, n);
		// allocate space
		a = (double*) malloc(sizeof(double) * n * n);
		b = (double*) malloc(sizeof(double) * n * n);
		c0 = (double*) malloc(sizeof(double) * n * n);
		c1 = (double*) malloc(sizeof(double) * n * n);
		// prepare data in range [0, 1)
		for (int i = 0; i < n * n; i++) {
			a[i] = (rand() % 100) / 100.0;
			b[i] = (rand() % 100) / 100.0;
		}

		// call dgemm0 as compare
		printf("Calling dgemm0.\n");
		clock_t start = clock();
		memset(c0, 0, sizeof(double) * n * n);
		dgemm0(a, b, c0, n);
		double elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));

		// call dgemm11
		printf("\nCalling dgemm11 - ijk.\n");
	    start = clock();		
		memset(c1, 0, sizeof(double) * n * n);
		dgemm11(a, b, c1, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
		printf("Maximum diff between dgemm0 and dgemm11: %.15f\n", verify(c0, c1, n));

		// call dgemm12
		printf("\nCalling dgemm12 - ikj.\n");
		start = clock();		
		memset(c1, 0, sizeof(double) * n * n);
		dgemm12(a, b, c1, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
		printf("Maximum diff between dgemm0 and dgemm12: %.15f\n", verify(c0, c1, n));

		// call dgemm11
		printf("\nCalling dgemm13 - jik.\n");
		start = clock();		
		memset(c1, 0, sizeof(double) * n * n);
		dgemm13(a, b, c1, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
		printf("Maximum diff between dgemm0 and dgemm13: %.15f\n", verify(c0, c1, n));

		// call dgemm11
		printf("\nCalling dgemm14 - jki.\n");
		start = clock();		
		memset(c1, 0, sizeof(double) * n * n);
		dgemm14(a, b, c1, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
		printf("Maximum diff between dgemm0 and dgemm14: %.15f\n", verify(c0, c1, n));

		// call dgemm11
		printf("\nCalling dgemm15 - kij.\n");
		start = clock();		
		memset(c1, 0, sizeof(double) * n * n);
		dgemm15(a, b, c1, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
		printf("Maximum diff between dgemm0 and dgemm15: %.15f\n", verify(c0, c1, n));

		// call dgemm16
		printf("\nCalling dgemm16 - kji.\n");
		start = clock();		
		memset(c1, 0, sizeof(double) * n * n);
		dgemm16(a, b, c1, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
		printf("Maximum diff between dgemm0 and dgemm16: %.15f\n", verify(c0, c1, n));


		// for blocking algorithms
		for (int j = 0; j < 5; j++) {
			printf("\nUsing blocking size: %d\n", blockSizes[j]);

			// call dgemm21
			printf("\nCalling dgemm21 - blocking ijk.\n");
			start = clock();		
			memset(c1, 0, sizeof(double) * n * n);
			dgemm21(a, b, c1, n, blockSizes[j]);
			elapse = clock() - start;
			printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
			printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
			printf("Maximum diff between dgemm0 and dgemm21: %.15f\n", verify(c0, c1, n));

			// call dgemm22
			printf("\nCalling dgemm22 - blocking ikj.\n");
			start = clock();		
			memset(c1, 0, sizeof(double) * n * n);
			dgemm22(a, b, c1, n, blockSizes[j]);
			elapse = clock() - start;
			printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
			printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
			printf("Maximum diff between dgemm0 and dgemm22: %.15f\n", verify(c0, c1, n));

			// call dgemm23
			printf("\nCalling dgemm23 - blocking jik.\n");
			start = clock();		
			memset(c1, 0, sizeof(double) * n * n);
			dgemm23(a, b, c1, n, blockSizes[j]);
			elapse = clock() - start;
			printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
			printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
			printf("Maximum diff between dgemm0 and dgemm23: %.15f\n", verify(c0, c1, n));

			// call dgemm24
			printf("\nCalling dgemm24 - blocking jki.\n");
			start = clock();		
			memset(c1, 0, sizeof(double) * n * n);
			dgemm24(a, b, c1, n, blockSizes[j]);
			elapse = clock() - start;
			printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
			printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
			printf("Maximum diff between dgemm0 and dgemm24: %.15f\n", verify(c0, c1, n));

			// call dgemm25
			printf("\nCalling dgemm25 - blocking kij.\n");
			start = clock();		
			memset(c1, 0, sizeof(double) * n * n);
			dgemm25(a, b, c1, n, blockSizes[j]);
			elapse = clock() - start;
			printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
			printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
			printf("Maximum diff between dgemm0 and dgemm25: %.15f\n", verify(c0, c1, n));

			// call dgemm26
			printf("\nCalling dgemm26 - blocking kji.\n");
			start = clock();		
			memset(c1, 0, sizeof(double) * n * n);
			dgemm26(a, b, c1, n, blockSizes[j]);
			elapse = clock() - start;
			printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
			printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));
			printf("Maximum diff between dgemm0 and dgemm26: %.15f\n", verify(c0, c1, n));

		}

		// release space
		free(a);
		free(b);
		free(c0);
		free(c1);
		printf("\n");
	}


	return 0;
}

// function dgemm0
void dgemm0 (double *a, double *b, double *c, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				c[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
}

// function dgemm11 - ijk
void dgemm11 (double *a, double *b, double *c, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			register double r = c[i * n + j];
			for (int k = 0; k < n; k++)
				r += a[i * n + k] * b[k * n + j];
			c[i * n + j] = r;
		}
	}
}

// function dgemm12 - ikj
void dgemm12 (double *a, double *b, double *c, int n) {
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < n; k++) {
			register double r = a[i * n + k];
			for (int j = 0; j < n; j++)
				c[i * n + j] += r * b[k * n + j];
		}
	}
}

// function dgemm13 - jik
void dgemm13 (double *a, double *b, double *c, int n) {
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++) {
			register double r = c[i * n + j];
			for (int k = 0; k < n; k++)
				r += a[i * n + k] * b[k * n + j];
			c[i * n + j] = r;
		}
	}
}

// function dgemm14 - jki
void dgemm14 (double *a, double *b, double *c, int n) {
	for (int j = 0; j < n; j++) {
		for (int k = 0; k < n; k++) {
			register double r = b[k * n + j];
			for (int i = 0; i < n; i++)
				c[i * n + j] += a[i * n + k] * r;
		}
	}
}

// function dgemm15 - kij
void dgemm15 (double *a, double *b, double *c, int n) {
	for (int k = 0; k < n; k++) {
		for (int i = 0; i < n; i++) {
			register double r = a[i * n + k];
			for (int j = 0; j < n; j++)
				c[i * n + j] += r * b[k * n + j];
		}
	}
}

// function dgemm16 - kji
void dgemm16 (double *a, double *b, double *c, int n) {
	for (int k = 0; k < n; k++) {
		for (int j = 0; j < n; j++) {
			register double r = b[k * n + j];
			for (int i = 0; i < n; i++)
				c[i * n + j] += a[i * n + k] * r;
		}
	}
}

// function dgemm21 - blocking ijk 
void dgemm21 (double *a, double *b, double *c, int n, int B) {
	for (int i = 0; i < n; i+=B) {
		for (int j = 0; j < n; j+=B) {
			for (int k = 0; k < n; k+=B) {
				// compute boundary of inner loop
				int ie = i + B < n? i + B : n;
				int je = j + B < n? j + B : n;
				int ke = k + B < n? k + B : n;
				for (int i1 = i; i1 < ie; i1++) {
					for (int j1 = j; j1 < je; j1++) {
						register double r = c[i1 * n + j1];
						for (int k1 = k; k1 < ke; k1++)
							r += a[i1 * n + k1] * b[k1 * n + j1];
						c[i1 * n + j1] = r;
					}
				}
			}
		}
	}
}

// function dgemm22 - blocking ikj 
void dgemm22 (double *a, double *b, double *c, int n, int B) {
	for (int i = 0; i < n; i+=B) {
		for (int k = 0; k < n; k+=B) {
			for (int j = 0; j < n; j+=B) {
				// compute boundary of inner loop
				int ie = i + B < n? i + B : n;
				int je = j + B < n? j + B : n;
				int ke = k + B < n? k + B : n;
				for (int i1 = i; i1 < ie; i1++) {
					for (int k1 = k; k1 < ke; k1++) {
						register double r = a[i1 * n + k1];
						for (int j1 = j; j1 < je; j1++)
							c[i1 * n + j1] += r * b[k1 * n + j1];
					}
				}
			}
		}
	}
}

// function dgemm23 - blocking jik 
void dgemm23 (double *a, double *b, double *c, int n, int B) {
	for (int j = 0; j < n; j+=B) {
		for (int i = 0; i < n; i+=B) {
			for (int k = 0; k < n; k+=B) {
				// compute boundary of inner loop
				int ie = i + B < n? i + B : n;
				int je = j + B < n? j + B : n;
				int ke = k + B < n? k + B : n;
				for (int j1 = j; j1 < je; j1++) {
					for (int i1 = i; i1 < ie; i1++) {
						register double r = c[i1 * n + j1];
						for (int k1 = k; k1 < ke; k1++)
							r += a[i1 * n + k1] * b[k1 * n + j1];
						c[i1 * n + j1] = r;
					}
				}
			}
		}
	}
}

// function dgemm24 - blocking jki 
void dgemm24 (double *a, double *b, double *c, int n, int B) {
	for (int j = 0; j < n; j+=B) {
		for (int k = 0; k < n; k+=B) {
			for (int i = 0; i < n; i+=B) {
				// compute boundary of inner loop
				int ie = i + B < n? i + B : n;
				int je = j + B < n? j + B : n;
				int ke = k + B < n? k + B : n;
				for (int j1 = j; j1 < je; j1++) {
					for (int k1 = k; k1 < ke; k1++) {
						register double r = b[k1 * n + j1];
						for (int i1 = i; i1 < ie; i1++)
							c[i1 * n + j1] += a[i1 * n + k1] * r;
					}
				}
			}
		}
	}
}

// function dgemm25 - blocking kij 
void dgemm25 (double *a, double *b, double *c, int n, int B) {
	for (int k = 0; k < n; k+=B) {
		for (int i = 0; i < n; i+=B) {
			for (int j = 0; j < n; j+=B) {
				// compute boundary of inner loop
				int ie = i + B < n? i + B : n;
				int je = j + B < n? j + B : n;
				int ke = k + B < n? k + B : n;
				for (int k1 = k; k1 < ke; k1++) {
					for (int i1 = i; i1 < ie; i1++) {
						register double r = a[i1 * n + k1];
						for (int j1 = j; j1 < je; j1++)
							c[i1 * n + j1] += r * b[k1 * n + j1];
					}
				}
			}
		}
	}
}

// function dgemm26 - blocking kji
void dgemm26 (double *a, double *b, double *c, int n, int B) {
	for (int k = 0; k < n; k+=B) {
		for (int j = 0; j < n; j+=B) {
			for (int i = 0; i < n; i+=B) {
				// compute boundary of inner loop
				int ie = i + B < n? i + B : n;
				int je = j + B < n? j + B : n;
				int ke = k + B < n? k + B : n;
				for (int k1 = k; k1 < ke; k1++) {
					for (int j1 = j; j1 < je; j1++) {
						register double r = b[k1 * n + j1];
						for (int i1 = i; i1 < ie; i1++)
							c[i1 * n + j1] += a[i1 * n + k1] * r;
					}
				}
			}
		}
	}
}


// function dgemm2
void dgemm2 (double *a, double *b, double *c, int n) {
	for (int i = 0; i < n; i+=2) {
		for (int j = 0; j < n; j+=2) {
			register double c00 = c[i * n + j];
			register double c01 = c[i * n + j + 1];
			register double c10 = c[(i + 1) * n + j];
			register double c11 = c[(i + 1) * n + j + 1];
			for (int k = 0; k < n; k+=2) {
				register double a00 = a[i * n + k];
				register double a01 = a[i * n + k + 1];
				register double a10 = a[(i + 1) * n + k];
				register double a11 = a[(i + 1) * n + k + 1];

				register double b00 = b[k * n + j];
				register double b01 = b[k * n + j + 1];
				register double b10 = b[(k + 1) * n + j];
				register double b11 = b[(k + 1) * n + j + 1];

				c00 += a00 * b00 + a01 * b10;
				c01 += a00 * b01 + a01 * b11;
				c10 += a10 * b00 + a11 * b10;
				c11 += a10 * b01 + a11 * b11;
			}
			c[i * n + j] = c00;
			c[i * n + j + 1] = c01;
			c[(i + 1) * n + j] = c10;
			c[(i + 1) * n + j + 1] = c11;
		}
	}
}

// function dgemm3
// compute 4 * 2 block in c at one time
void dgemm3 (double *a, double *b, double *c, int n) {
	for (int i = 0; i < n; i+=4) {
		for (int j = 0; j < n; j+=2) {
			register double c00 = c[i * n + j];
			register double c01 = c[i * n + j + 1];
			register double c10 = c[(i + 1) * n + j];
			register double c11 = c[(i + 1) * n + j + 1];
			register double c20 = c[(i + 2) * n + j];
			register double c21 = c[(i + 2) * n + j + 1];
			register double c30 = c[(i + 3) * n + j];
			register double c31 = c[(i + 3) * n + j + 1];
			for (int k = 0; k < n; k+=2) {
				register double a0 = a[i * n + k];
				register double a1 = a[(i + 1) * n + k];
				register double a2 = a[(i + 2) * n + k];
				register double a3 = a[(i + 3) * n + k];

				register double b0 = b[k * n + j];
				register double b1 = b[k * n + j + 1];

				//printf("i: %d, j: %d, k: %d, a: %f %f %f %f, b: %f %f\n", i, j, k, a0, a1, a2, a3, b0, b1);

				c00 += a0 * b0;
				c01 += a0 * b1;
				c10 += a1 * b0;
				c11 += a1 * b1;
				c20 += a2 * b0;
				c21 += a2 * b1;
				c30 += a3 * b0;
				c31 += a3 * b1;
				// get second group of elements
				a0 = a[i * n + k + 1];
				a1 = a[(i + 1) * n + k + 1];
				a2 = a[(i + 2) * n + k + 1];
				a3 = a[(i + 3) * n + k + 1];

				b0 = b[(k + 1) * n + j];
				b1 = b[(k + 1) * n + j + 1];

				//printf("i: %d, j: %d, k: %d, a: %f %f %f %f, b: %f %f\n", i, j, k, a0, a1, a2, a3, b0, b1);

				c00 += a0 * b0;
				c01 += a0 * b1;
				c10 += a1 * b0;
				c11 += a1 * b1;
				c20 += a2 * b0;
				c21 += a2 * b1;
				c30 += a3 * b0;
				c31 += a3 * b1;


			}
			c[i * n + j] = c00;
			c[i * n + j + 1] = c01;
			c[(i + 1) * n + j] = c10;
			c[(i + 1) * n + j + 1] = c11;
			c[(i + 2) * n + j] = c20;
			c[(i + 2) * n + j + 1] = c21;
			c[(i + 3) * n + j] = c30;
			c[(i + 3) * n + j + 1] = c31;
		}
	}
}

// get the maximum diff
double verify (double *c0, double *c1, int n) {
	double diff = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double tmp = c0[i * n + j] - c1[i * n + j];
			tmp = tmp > 0? tmp : -1 * tmp;
			if (tmp > diff) diff = tmp;
		}
	}
	return diff;
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