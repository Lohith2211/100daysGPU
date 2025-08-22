#include <iostream>
#include <cublas_v2.h>

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    for(int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_a, *d_b;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));

    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;

    cublasSaxpy(handle, N, &alpha, d_a, 1, d_b, 1);

    cudaMemcpy(C, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cublasDestroy(handle);

    return 0;
}