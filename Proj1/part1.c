#include "stdio.h"
#include "stdlib.h"
#include "time.h"


void dgemm0 (double *a, double *b, double *c, int n);
void dgemm1 (double *a, double *b, double *c, int n);
double verify (double *, double *, int n);

int main () {
	srand(211);
	int sizes[] = {64, 128, 512, 1024, 2048};
	double *a, *b, *c0, *c1; // c0 for result of degmm0, c1 for result of degmm1
	for (int i = 0; i < 5; i++) {
		// get size of this iteration
		int n = sizes[i]; // length or width
		printf("Size of matrix: %d * %d\n", n, n);
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
		// call dgemm0
		printf("Calling dgemm0.\n");
		clock_t start = clock();
		dgemm0(a, b, c0, n);
		double elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n / (elapse * 1000000000 / CLOCKS_PER_SEC));

		// call dgemm1
		printf("Calling dgemm1.\n");
		start = clock();
		dgemm1(a, b, c1, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n / (elapse * 1000000000 / CLOCKS_PER_SEC));
		// verify results
		printf("Maximum diff: %.15f\n", verify(c0, c1, n));
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

// function dgemm1
void dgemm1 (double *a, double *b, double *c, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			register double r = c[i * n + j];
			for (int k = 0; k < n; k++)
				r += a[i * n + k] * b[k * n + j];
			c[i * n + j] = r;
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