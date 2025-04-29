const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
   // insert your code here: implementation of vectorized vector-matrix multiply

   #pragma omp parallel for // compiler for vectorization from P&H Reading
    for (int i = 0; i < n; i++) {
      double temp = 0.0;
        
      // Let compiler auto-vectorize this inner loop
      for (int j = 0; j < n; j++) {
          temp += A[i * n + j] * x[j];
      }
      y[i] += temp;
    }
}
