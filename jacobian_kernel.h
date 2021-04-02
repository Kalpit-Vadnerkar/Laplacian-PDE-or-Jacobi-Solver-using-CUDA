#include <cuda_runtime.h> 
#include <cuda.h> 
#include "utils.h"

/*
 * The launcher for your kernels. 
 * This is a single entry point and 
 * all arrays **MUST** be pre-allocated 
 * on device. you must implement all other 
 * kernels in the respective files.
 * */ 


void launch_jacobian(float* in_mat, float* out_mat, const int nrows, const int ncols);
