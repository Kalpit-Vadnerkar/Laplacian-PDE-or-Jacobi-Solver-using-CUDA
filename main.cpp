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


bool are_similar(float** a, float** b,
        const int nrow, const int ncol)
{
    float error = 0;
    for (int r = 0; r < nrow; ++r)
    {
        for (int c = 0; c < ncol; ++c)
        {
            error += (float)fabs(a[r][c] - b[r][c]);
        }
    }
    if (error > 0.000001f)
    {
      return false;
    }

    return true;
}

void serial(float** h_in, const int numRow, const int numCol)
{
    float** temp = new float*[numRow]();
    for (int i = 0; i < numRow; ++i)
    {
        temp[i] = new float[numCol]();
    }

    for (int i = 0; i < 100; ++i)
    {
        for (int r = 1; r < numRow -1; ++r)
        {   
            for (int c = 1; c < numCol -1; ++c)
            {
                temp[r][c] = (h_in[r - 1][c] + h_in[r][c - 1] + h_in[r][c + 1] + h_in[r + 1][c]) / 4;
            }
        }
        
        
        if (are_similar(h_in, temp, numRow, numCol))
        {
            break;
        }

        
        for (int r = 1; r < numRow -1; ++r)
        {
            memcpy(h_in[r], temp[r], numCol *sizeof(float));
        }
    }
}

void check_results(float** serial_mat, float* gpu_mat, const int rows, const int cols)
{
    float e = 0;
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int index = r*cols + c;
            e = fabs(serial_mat[r][c] - gpu_mat[index]);
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
    srand(76465);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;

    int h_nrows, h_ncols;
    float** h_in, *h_result;
    float* d_in, *d_out, *d_error;


    // add 2 for padding of zeros
    h_nrows = std::stoi(argv[1], nullptr);
    h_ncols = std::stoi(argv[2], nullptr);
    
    
    size_t size = h_nrows*h_ncols*sizeof(float);

    h_in = new float*[h_nrows]();
    for (int i = 0; i < h_nrows; ++i)
    {
        h_in[i] = new float[h_ncols]();
    }
    
    for (int r = 1; r < h_nrows - 1; ++r)
    {
        for (int c = 1; c < h_ncols - 1; ++c)
        {
            h_in[r][c] = (float)(rand() % 255 - 100);
        }
    }

    // allocate device memory
    checkCudaErrors(cudaMalloc((void**)&d_in, size));
    checkCudaErrors(cudaMalloc((void**)&d_out, size));
    checkCudaErrors(cudaMalloc((void**)&d_error, sizeof(float)));

    // setup device memory
    checkCudaErrors(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_out, h_in, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_error, 0.0f, sizeof(float)));


    auto start = high_resolution_clock::now();
    serial(h_in, h_nrows, h_ncols);
    auto end = high_resolution_clock::now();
    auto serial_dur = duration_cast<microseconds>(end - start);

    std::cout << "serial time: " << serial_dur.count() << "microseconds" << std::endl;

    
    // call kernel
    launch_jacobian(d_in, d_out, h_nrows, h_ncols, d_error);

    // get results from device
    h_result = new float[size];

    checkCudaErrors(cudaMemcpy(h_result, d_out, size, cudaMemcpyDeviceToHost));

   
    check_results(h_in, h_result, h_nrows, h_ncols);
    printf("PASSED!!!!!!\n");

    // cleanup
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}



