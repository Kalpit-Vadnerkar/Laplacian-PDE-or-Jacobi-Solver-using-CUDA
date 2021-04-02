#include "jacobian_kernel.h"
#include "constants.h"

#define BLOCK 32

__global__ 
void laplacePDE(float *d_in, float *d_temp, int numRows, int numCols, float *d_error){

 /*
   Your kernel here: Make sure to check for boundary conditions
  */

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int Offset = y * numCols + x;

  if (x < numCols - 1 && x > 0 && y < numRows - 1 && y > 0){
    d_temp[Offset] = (d_in[(y - 1) * numCols + x] + d_in[y * numCols + x - 1] + d_in[y * numCols + x + 1] + d_in[(y + 1) * numCols + x]) / 4;
  }
  __syncthreads();
  if (x < numCols && y < numRows){
    d_error += abs(d_temp[Offset] - d_in[Offset]);
    __syncthreads();
    d_in[Offset] = d_temp[Offset];
  }

}




void launch_jacobian(float* d_in, float* d_temp, const int numRows, const int numCols, float* d_error){
    // configure launch params here 
    
    dim3 block(BLOCK, BLOCK, 1);
    dim3 grid((numCols-1)/BLOCK + 1, (numRows-1)/BLOCK + 1, 1);
    for (int i = 0; i < NUM_ITER; ++i){
        d_error = 0;
        laplacePDE<<<grid,block>>>(d_in, d_temp, numRows, numCols, d_error);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        if (d_error < EPSILON){
            break;        
        }
    }
}





