#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void mish_kernel(float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float softplus = logf(1 + expf(val)); 
        y[idx] = val * tanhf(softplus);       
    }
}

void mish_cuda(float* d_x, float* d_y, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    mish_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, size);
    cudaDeviceSynchronize();
}

int main() {
    const int size = 10; 
    size_t bytes = size * sizeof(float);

    float* h_x = new float[size];
    float* h_y = new float[size];

    for (int i = 0; i < size; i++) {
        h_x[i] = (float)(i - 5);  
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_y, size * sizeof(float));

    cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    mish_cuda<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, size);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Input\tMish Output\n";
    for (int i = 0; i < size; i++) {
        std::cout << "Mish(" << h_x[i] << ") = " << h_y[i] << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}