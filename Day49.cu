#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define N 4

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void matVecMul(const float *A, const float *x, float *y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[row * n + j] * x[j];
        }
        y[row] = sum;
    }
}

__global__ void vecAdd(float *y, const float *x, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += alpha * x[i];
    }
}

__global__ void vecSub(float *y, const float *x, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] -= alpha * x[i];
    }
}

__global__ void vecScale(float *x, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= beta;
    }
}

__global__ void dotProduct(const float *a, const float *b, float *result, int n) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    while (i < n) {
        temp += a[i] * b[i];
        i += blockDim.x * gridDim.x;
    }
    cache[tid] = temp;
    __syncthreads();

    int blockSize = blockDim.x;
    while (blockSize > 1) {
        int halfPoint = blockSize / 2;
        if (tid < halfPoint) {
            cache[tid] += cache[tid + halfPoint];
        }
        __syncthreads();
        blockSize = halfPoint;
    }
    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}

int main() {
    const int n = N;
    const int matrixSize = n * n * sizeof(float);
    const int vectorSize = n * sizeof(float);

    float h_A[N * N] = {
         4, 1, 0, 0,
         1, 3, 1, 0,
         0, 1, 2, 1,
         0, 0, 1, 1
    };
    float h_b[N] = {15, 10, 10, 10};
    float h_x[N] = {0}; 

    float *d_A, *d_x, *d_b, *d_r, *d_p, *d_Ap;
    CUDA_CHECK(cudaMalloc(&d_A, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_x, vectorSize));
    CUDA_CHECK(cudaMalloc(&d_b, vectorSize));
    CUDA_CHECK(cudaMalloc(&d_r, vectorSize));
    CUDA_CHECK(cudaMalloc(&d_p, vectorSize));
    CUDA_CHECK(cudaMalloc(&d_Ap, vectorSize));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, vectorSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, vectorSize, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_r, d_b, vectorSize, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, d_r, vectorSize, cudaMemcpyDeviceToDevice));

    float *d_dot;
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(float)));

    float rdotr = 0.0f, new_rdotr = 0.0f;

    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));
    dotProduct<<<1, 256>>>(d_r, d_r, d_dot, n);
    CUDA_CHECK(cudaMemcpy(&rdotr, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    int max_iter = 1000;
    float tol = 1e-6f;
    int k = 0;

    while (sqrt(rdotr) > tol && k < max_iter) {
   
        matVecMul<<<(n + 255) / 256, 256>>>(d_A, d_p, d_Ap, n);

        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));
        dotProduct<<<1, 256>>>(d_p, d_Ap, d_dot, n);
        float pAp = 0.0f;
        CUDA_CHECK(cudaMemcpy(&pAp, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

        float alpha = rdotr / pAp;

        vecAdd<<<(n + 255) / 256, 256>>>(d_x, d_p, alpha, n);

        vecSub<<<(n + 255) / 256, 256>>>(d_r, d_Ap, alpha, n);

        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));
        dotProduct<<<1, 256>>>(d_r, d_r, d_dot, n);
        CUDA_CHECK(cudaMemcpy(&new_rdotr, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

        if (sqrt(new_rdotr) < tol) {
            break;
        }

        float beta = new_rdotr / rdotr;

        vecScale<<<(n + 255) / 256, 256>>>(d_p, beta, n);
        
        vecAdd<<<(n + 255) / 256, 256>>>(d_p, d_r, 1.0f, n);

        rdotr = new_rdotr;
        k++;
    }

    CUDA_CHECK(cudaMemcpy(h_x, d_x, vectorSize, cudaMemcpyDeviceToHost));
    std::cout << "Conjugate Gradient converged in " << k << " iterations." << std::endl;
    std::cout << "Solution x:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << h_x[i] << std::endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_dot));

    return 0;
}