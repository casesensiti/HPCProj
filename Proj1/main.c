#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"


void dgemm0 (double *a, double *b, double *c, int n);
void dgemm1 (double *a, double *b, double *c, int n);
void dgemm2 (double *a, double *b, double *c, int n);
void dgemm3 (double *a, double *b, double *c, int n);
double verify (double *, double *, int n);
void dump (double *c, int n);

int main () {
	srand(clock());
	int sizes[] = {64, 128, 512, 1024, 2048};
	double *a, *b, *c0, *c1, *c2, *c3; // c0 for result of degmm0, c1 for result of degmm1
	for (int i = 0; i < 5; i++) {
		// get size of this iteration
		int n = sizes[i]; // length or width
		printf("Size of matrix: %d * %d\n", n, n);
		// allocate space
		a = (double*) malloc(sizeof(double) * n * n);
		b = (double*) malloc(sizeof(double) * n * n);
		c0 = (double*) malloc(sizeof(double) * n * n);
		c1 = (double*) malloc(sizeof(double) * n * n);
		c2 = (double*) malloc(sizeof(double) * n * n);
		c3 = (double*) malloc(sizeof(double) * n * n);
		// prepare data in range [0, 1)
		for (int i = 0; i < n * n; i++) {
			a[i] = (rand() % 100) / 100.0;
			b[i] = (rand() % 100) / 100.0;
		}

		// clear result buffers
		memset(c0, 0, sizeof(double) * n * n);
		memset(c1, 0, sizeof(double) * n * n);
		memset(c2, 0, sizeof(double) * n * n);
		memset(c3, 0, sizeof(double) * n * n);

		// call dgemm0
		printf("Calling dgemm0.\n");
		clock_t start = clock();
		dgemm0(a, b, c0, n);
		double elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC / (elapse * 1000000000));

		// call dgemm1
		printf("Calling dgemm1.\n");
		start = clock();
		dgemm1(a, b, c1, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC/ (elapse * 1000000000));
		
		// call dgemm2
		printf("Calling dgemm2.\n");
		start = clock();
		dgemm2(a, b, c2, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC/ (elapse * 1000000000));
				
		// call dgemm3
		printf("Calling dgemm3.\n");
		start = clock();
		dgemm3(a, b, c3, n);
		elapse = clock() - start;
		printf("Time used: %fms\n", 1000 * elapse / CLOCKS_PER_SEC);
		printf("Performance (in Gflops): %f\n", 2.0 * n * n * n * CLOCKS_PER_SEC/ (elapse * 1000000000));
		
		// verify results
		printf("Maximum diff between dgemm0 and dgemm1: %.15f\n", verify(c0, c1, n));
		printf("Maximum diff between dgemm0 and dgemm2: %.15f\n", verify(c0, c2, n));
		printf("Maximum diff between dgemm0 and dgemm3: %.15f\n", verify(c0, c3, n));

		// release space
		free(a);
		free(b);
		free(c0);
		free(c1);
		free(c2);
		free(c3);
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