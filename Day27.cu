#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define ETA 0.5f 

#define EUCLIDEAN         0  
#define NEGATIVE_ENTROPY  1  
#define LOG_BARRIER       2  

__global__ void mirror_descent(float *x, float *grad, float eta, int mirror_map, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float new_x = x[i];

    switch (mirror_map) {
        case EUCLIDEAN:
            new_x = x[i] - eta * grad[i];
            break;

        case NEGATIVE_ENTROPY:
            new_x = x[i] * expf(-eta * grad[i]); 
            break;

        case LOG_BARRIER:
            new_x = x[i] / (1.0f + eta * grad[i]);
            break;

        default:
            new_x = x[i]; 
    }

    x[i] = new_x;
}

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(result));
        exit(-1);
    }
}

int main() {
    float *x, *grad, *d_x, *d_grad;
    int mirror_map = NEGATIVE_ENTROPY; 

    x = (float*)malloc(N * sizeof(float));
    grad = (float*)malloc(N * sizeof(float));
    checkCuda(cudaMalloc(&d_x, N * sizeof(float)), "Alloc d_x");
    checkCuda(cudaMalloc(&d_grad, N * sizeof(float)), "Alloc d_grad");

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;  
        grad[i] = 0.5f * i; 
    }

    checkCuda(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy x -> d_x");
    checkCuda(cudaMemcpy(d_grad, grad, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy grad -> d_grad");

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    mirror_descent<<<numBlocks, blockSize>>>(d_x, d_grad, ETA, mirror_map, N);
    checkCuda(cudaDeviceSynchronize(), "Kernel execution");

    checkCuda(cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy d_x -> x");

    for (int i = 0; i < 10; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    free(x);
    free(grad);
    cudaFree(d_x);
    cudaFree(d_grad);

    return 0;
}