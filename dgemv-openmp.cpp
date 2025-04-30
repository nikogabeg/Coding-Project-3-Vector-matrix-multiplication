// Niko Galedo
// CSC 656-01
//
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

const char* dgemv_desc = "OpenMP dgemv.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */

void my_dgemv(int n, double* A, double* x, double* y) {

   // use #pragma omp parallel for to parallelize row computations from lecture page 51
   #pragma omp parallel for
   for (int i = 0; i < n; i++) {
      double temp = 0.0;
      for (int j = 0; j < n; j++) {
          temp += A[i * n + j] * x[j]; // Row-major access (Page 9)
      }
      y[i] += temp; // No atomic/critical needed (distinct per thread, Page 51)
  }
   // insert your dgemv code here. you may need to create additional parallel regions,
   // and you will want to comment out the above parallel code block that prints out
   // nthreads and thread_id so as to not taint your timings

}

