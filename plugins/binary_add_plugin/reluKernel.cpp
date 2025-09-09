#include <hip/hip_runtime.h>

__global__ void reluKernel(float* output, const float* input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" void launchRelu(float* output, const float* input, size_t size, hipStream_t stream) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(reluKernel, 
                       dim3(grid_size), 
                       dim3(block_size), 
                       0, 
                       stream, 
                       output, 
                       input, 
                       size);
}