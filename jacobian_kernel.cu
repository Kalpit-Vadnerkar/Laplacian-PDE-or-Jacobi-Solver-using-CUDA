#include "jacobian_kernel.h"

#define BLOCK 32

__global__ 
void im2Gray(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols){

 /*
   Your kernel here: Make sure to check for boundary conditions
  */

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  

  if (x < numCols && y < numRows){
    int grayOffset = y * numCols + x;
    unsigned char r = d_in[grayOffset].x; 
    unsigned char g = d_in[grayOffset].y;
    unsigned char b = d_in[grayOffset].z;
    d_grey[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
  } 
}




void launch_jacobian(float* in_mat, float* out_mat, const int numRows, const int numCols){
    // configure launch params here 
    
    dim3 block(BLOCK, BLOCK, 1);
    dim3 grid((numCols-1)/BLOCK + 1, (numRows-1)/BLOCK + 1, 1);
    
    im2Gray<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
}





