#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert> 
#include <string> 
#include <cmath>
#include <chrono>
#include <cstring>

#include "utils.h"
#include "jacobian_kernel.h"
#include "constants.h"

void fill_mat(float** mat, const int nrows, const int ncols)
{
   for (int r = 1; r < nrows-1; ++r)
   {
       for (int c = 1; c < ncols-1; ++c)
       {
           // fill in matrix with values between -100 and 100
           mat[r][c] = (float)(rand() % 201 - 100);
       }
   }
}


bool compare_mats(float** const mat1, float** const mat2,
        const int nrow, const int ncol)
{
    float error = 0;
    for (int r = 0; r < nrow; ++r)
    {
        for (int c = 0; c < ncol; ++c)
        {
            error += (float)fabs(mat1[r][c] - mat2[r][c]);
        }
    }
    if (error > EPSILON)
    {
      return false;
    }

    return true;
}

void serial_solver(float** mat, const int nrow, const int ncol)
{
    float** temp_mat = new float*[nrow]();
    for (int i = 0; i < nrow; ++i)
    {
        temp_mat[i] = new float[ncol]();
    }

    for (int i = 0; i < NUM_ITER; ++i)
    {
        // calculate the next iteration of the matrix
        for (int r = 1; r < nrow-1; ++r)
        {   
            float sum;
            float avg;
            int nsummed = 4;

            for (int c = 1; c < ncol-1; ++c)
            {
                sum = mat[r-1][c] + mat[r][c-1] + mat[r][c+1] + mat[r+1][c];
                avg = (float)sum / (float)nsummed;
                temp_mat[r][c] = avg;
            }
        }
        
        // check if the calculation has converged enough
        if (compare_mats(mat, temp_mat, nrow, ncol))
        {
            break;
        }

        // copy matrix for next iteration
        for (int r = 1; r < nrow-1; ++r)
        {
            memcpy(mat[r], temp_mat[r], ncol*sizeof(float));
        }
    }

    for (int i = 0; i < nrow; ++i)
    {
        delete[] temp_mat[i];
    }
    delete[] temp_mat;
}

void check_results(float** correct_mat, float* test_mat, const int rows, const int cols)
{
    float e = 0;
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int index = r*cols + c;
            e = fabs(correct_mat[r][c] - test_mat[index]);        
        }
    }
    if (e < 0.001)
    {
      printf("FAILED: matrices are different \n");
      exit(0);
    }
}

int main(int argc, char const *argv[]) 
{
    srand(40698);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;

    int h_nrows, h_ncols;
    int d_nrows, d_ncols;
    float** h_in_mat, **h_out_mat;
    float* d_in_mat, *d_out_mat, *d_result_mat, *d_error;

    if (argc != 3)
    {
        fprintf(stderr, "ERROR: run as ./{program_name} {num_rows} {num_cols}\n");
        exit(0);
    }

    // add 2 for padding of zeros
    h_nrows = std::stoi(argv[1], nullptr) + 2;
    h_ncols = std::stoi(argv[2], nullptr) + 2;
    d_nrows = h_nrows;
    d_ncols = h_ncols;
    
    size_t mat_size = h_nrows*h_ncols*sizeof(float);

    auto start1 = high_resolution_clock::now();

    // allocate the host matrices
    h_in_mat = new float*[h_nrows]();
    h_out_mat = new float*[h_nrows]();
    for (int i = 0; i < h_nrows; ++i)
    {
        h_in_mat[i] = new float[h_ncols]();
        h_out_mat[i] = new float[h_ncols]();
    }
    auto end1 = high_resolution_clock::now();

    // populate the matrix with random values
    fill_mat(h_in_mat, h_nrows, h_ncols);

    // copy memory for the serial solver
    for (int i = 0; i < h_nrows; ++i)
    {
        memcpy(h_out_mat[i], h_in_mat[i], h_ncols*sizeof(float));
    }

    auto start2 = high_resolution_clock::now();
    serial_solver(h_out_mat, h_nrows, h_ncols);
    auto end2 = high_resolution_clock::now();

    auto serial_dur = duration_cast<microseconds>(end2-start2);

    std::cout << "serial time: " << serial_dur.count() << "microseconds" << std::endl;

    // allocate device memory
    checkCudaErrors(cudaMalloc((void**)&d_in_mat, mat_size));
    checkCudaErrors(cudaMalloc((void**)&d_out_mat, mat_size));
    checkCudaErrors(cudaMalloc((void**)&d_error, sizeof(float)));

    // setup device memory
    checkCudaErrors(cudaMemcpy(d_in_mat, h_in_mat, mat_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_out_mat, h_in_mat, mat_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_error, 0.0f, sizeof(float)));

    // call kernel
    launch_jacobian(d_in_mat, d_out_mat, d_nrows, d_ncols, d_error);

    // get results from device
    d_result_mat = new float[mat_size];

    checkCudaErrors(cudaMemcpy(d_result_mat, d_out_mat, mat_size, cudaMemcpyDeviceToHost));

    std::cout << std::endl;

    std::cout << std::endl;
    

    check_results(h_out_mat, d_result_mat, h_nrows, h_ncols);
    printf("TEST PASSED!!!!!!\n");

    // cleanup
    cudaFree(d_in_mat);
    cudaFree(d_out_mat);

    for (int i = 0; i < h_nrows; ++i)
    {
        delete[] h_in_mat[i];
        delete[] h_out_mat[i];
    }
    delete[] h_in_mat;
    delete[] h_out_mat;

    delete[] d_result_mat;

    return 0;
}



