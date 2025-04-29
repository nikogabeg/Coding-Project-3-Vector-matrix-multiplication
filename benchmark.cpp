// Niko Galedo
// CSC 656-01
//
//
// (C) 2021, E. Wes Bethel
// benchmark-* harness for running different versions of vector-matrix multiply
//    over different problem sizes
//
// usage: no command line arguments
// set problem sizes in the code below

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cmath> // For: fabs

#include "cblas.h"
#include <string.h>

// external definitions for mmul's
extern void my_dgemv(int, double*, double*, double *);
extern const char* dgemv_desc;

void reference_dgemv(int n, double* A, double* x, double *y) {
   double alpha=1.0, beta=1.0;
   int lda=n, incx=1, incy=1;
    // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A, n, B, n, 1., C, n);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, alpha, A, lda, x, incx, beta, y, incy);
}

void fill(double* p, int n) {
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        p[i] = 2 * dis(gen) - 1;
}

bool check_accuracy(double *A, double *Anot, int nvalues)
{
  double eps = 1e-5;
  for (size_t i = 0; i < nvalues; i++) 
  {
    if (fabsf(A[i] - Anot[i]) > eps) {
       return false;
    }
  }
  return true;
}


/* The benchmarking program */
int main(int argc, char** argv) 
{
    std::cout << "Description:\t" << dgemv_desc << std::endl << std::endl;

    std::cout << std::fixed << std::setprecision(5);

    // we purposefully run the smallest problem twice so as to "condition"
    // BLAS. For timing purposes, ignore the timing of the first problem size
    std::vector<int> test_sizes{1024, 1024, 2048, 4096, 8192, 16384};

    int n_problems = test_sizes.size();

    // preallocate memory buffers for all problems: assume the last number in test_sizes is the largest
     // Theoretical peak bandwidth for Perlmutter CPU nodes (200 GB/s)
     const double peak_bandwidth = 200.0 * 1024 * 1024 * 1024; // Convert to bytes/s

    // allocate memory for 2 NxN matrices and 4 Nx1 vectors

    int max_size = test_sizes[n_problems-1];

    std::vector<double> buf(2 * max_size * max_size + 4 * max_size);
    double* A = buf.data() + 0;
    double* Acopy = A + max_size * max_size;
    double* X = Acopy + max_size * max_size;
    double* Xcopy = X + max_size;
    double* Y = Xcopy + max_size;
    double* Ycopy = Y + max_size;

           // load up matrics with some random numbers

    // Skip first iteration flag (since first 1024 is for conditioning)
    bool skip_first = true;       
    /* For each test size */
    for (int n : test_sizes) 
    {
        printf("Working on problem size N=%d \n", n);

        fill(A, n * n);
        fill(X, n );
        fill(Y, n );

        // make copies of A, B, C for use in verification of results
        memcpy(Acopy, A, sizeof(double)*n*n);
        memcpy(Xcopy, X, sizeof(double)*n);
        memcpy(Ycopy, Y, sizeof(double)*n);

        // insert start timer code here
        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        // now invoke the cblas method to compute the matrix-vector multiplye
        reference_dgemv(n, Acopy, Xcopy, Ycopy);
        

        // insert end timer code here, and print out the elapsed time for this problem size
        // End timer and calculate elapsed time
        auto end_time = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end_time - start_time;

        // Skip the first run (conditioning run)
        if (skip_first) {
          skip_first = false;
          continue;
        }

        // Calculate metrics
        double flops = 2.0 * n * n; // 2n² FLOPs
        double mflops = (flops / 1e6) / elapsed.count();

        // Memory accesses: read A (n²), read X (n), read/write Y (n)
        double bytes_accessed = (n * n + 2 * n) * sizeof(double);
        double bandwidth_used = bytes_accessed / elapsed.count();
        double bandwidth_utilization = (bandwidth_used / peak_bandwidth) * 100.0;

        
        
        // Print results
        std::cout << "  Time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "  MFLOP/s: " << mflops << std::endl;
        std::cout << "  Memory bandwidth used: " << bandwidth_used / (1024*1024*1024) << " GB/s" << std::endl;
        std::cout << "  % of peak bandwidth: " << bandwidth_utilization << "%" << std::endl;

        // call the method to do the work
        my_dgemv(n, A, X, Y); 

        // compare your result with that computed by BLAS
        if (check_accuracy(Ycopy, Y, n) == false)
           printf(" Error: your answer is not the same as that computed by BLAS. \n");
    
    } // end loop over problem sizes

    return 0;
}

// EOF
