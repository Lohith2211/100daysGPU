#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void adaHessianUpdateKernel(
    float* theta,                
    const float* grad,           
    const float* gradPerturbed,  
    float* m,                    
    float* v,                    
    const float lr,              
    const float beta1,           
    const float beta2,           
    const float epsilon,         
    const float delta,           
    int N                        
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
      
        float h_diag = (gradPerturbed[idx] - grad[idx]) / delta;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * (h_diag * h_diag);
        
        theta[idx] -= lr * m[idx] / (sqrtf(v[idx]) + epsilon);
    }
}

int main() {
    const int N = 10;             
    const int bytes = N * sizeof(float);
    const float lr = 0.01f;         
    const float beta1 = 0.9f;   
    const float beta2 = 0.999f;     
    const float epsilon = 1e-7f;    
    const float delta = 1e-4f;      

    float h_theta[N], h_grad[N], h_gradPerturbed[N], h_m[N], h_v[N];
    
    for (int i = 0; i < N; i++) {
        h_theta[i] = 1.0f;          
        h_grad[i] = 0.1f;           
        h_gradPerturbed[i] = 0.1f + 0.001f * i;
        h_m[i] = 0.0f;             
        h_v[i] = 0.0f;             
    }

    float *d_theta, *d_grad, *d_gradPerturbed, *d_m, *d_v;
    cudaMalloc((void**)&d_theta, bytes);
    cudaMalloc((void**)&d_grad, bytes);
    cudaMalloc((void**)&d_gradPerturbed, bytes);
    cudaMalloc((void**)&d_m, bytes);
    cudaMalloc((void**)&d_v, bytes);

    cudaMemcpy(d_theta, h_theta, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradPerturbed, h_gradPerturbed, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    adaHessianUpdateKernel<<<gridSize, blockSize>>>(
        d_theta, d_grad, d_gradPerturbed, d_m, d_v,
        lr, beta1, beta2, epsilon, delta, N
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_theta, d_theta, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, d_m, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);

    printf("Updated theta values:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_theta[i]);
    }
    printf("\n");

    cudaFree(d_theta);
    cudaFree(d_grad);
    cudaFree(d_gradPerturbed);
    cudaFree(d_m);
    cudaFree(d_v);

    return 0;
}
