#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void dotProductKernel(const double* a, const double* b, double* result, int n) {
    __shared__ double cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    double temp = 0.0;
    while(tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if(cacheIndex < stride)
            cache[cacheIndex] += cache[cacheIndex + stride];
        __syncthreads();
    }
    if(cacheIndex == 0)
        atomicAdd(result, cache[0]);
}

__global__ void vectorSubKernel(double* a, const double* b, double scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        a[idx] -= scalar * b[idx];
    }
}

__global__ void vectorScaleKernel(double* a, double scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        a[idx] *= scalar;
    }
}

__global__ void vectorAddKernel(double* a, const double* b, double scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        a[idx] += scalar * b[idx];
    }
}

double gpuDot(const double* d_a, const double* d_b, int n) {
    double h_result = 0.0;
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    dotProductKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

int main() {
    const int n = 1024;      
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    double *h_x = (double*)malloc(n * sizeof(double));
    double *h_grad = (double*)malloc(n * sizeof(double));
    double *h_s = (double*)malloc(n * sizeof(double)); 
    double *h_y = (double*)malloc(n * sizeof(double)); 

    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0;     
        h_grad[i] = 0.5;    
        h_s[i] = 0.1;      
        h_y[i] = 0.2;       
    }

    double *d_x, *d_grad, *d_s, *d_y, *d_q, *d_r;
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_grad, n * sizeof(double));
    cudaMalloc(&d_s, n * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMalloc(&d_q, n * sizeof(double));
    cudaMalloc(&d_r, n * sizeof(double));

    cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_q, d_grad, n * sizeof(double), cudaMemcpyDeviceToDevice);

    double dot_sy = gpuDot(d_s, d_y, n);
    double rho = 1.0 / dot_sy;

    double dot_sq = gpuDot(d_s, d_q, n);
    double alpha = rho * dot_sq;

    vectorSubKernel<<<gridSize, blockSize>>>(d_q, d_y, alpha, n);

    double dot_yy = gpuDot(d_y, d_y, n);
    double H0 = dot_sy / dot_yy;

    cudaMemcpy(d_r, d_q, n * sizeof(double), cudaMemcpyDeviceToDevice);
    vectorScaleKernel<<<gridSize, blockSize>>>(d_r, H0, n);

    double dot_yr = gpuDot(d_y, d_r, n);
    double beta = rho * dot_yr;

    double scalar = (alpha - beta);
    vectorAddKernel<<<gridSize, blockSize>>>(d_r, d_s, scalar, n);

    vectorScaleKernel<<<gridSize, blockSize>>>(d_r, -1.0, n);

    double step = 0.1;  
    vectorAddKernel<<<gridSize, blockSize>>>(d_x, d_r, step, n);

    cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Updated parameters (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("x[%d] = %f\n", i, h_x[i]);
    }

    cudaFree(d_x);
    cudaFree(d_grad);
    cudaFree(d_s);
    cudaFree(d_y);
    cudaFree(d_q);
    cudaFree(d_r);

    free(h_x);
    free(h_grad);
    free(h_s);
    free(h_y);

    return 0;
}