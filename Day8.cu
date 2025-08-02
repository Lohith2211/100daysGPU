#define BLOCK_SIZE 32
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scanBrentKung(float* input, float* output, int size) {
    __shared__ float temp[BLOCK_SIZE]; 
    int tid = threadIdx.x;
    int idx = 2 * blockIdx.x * blockDim.x + tid;

    if (idx < size)
        temp[tid] = input[idx];
    if (idx + blockDim.x < size)
        temp[tid + blockDim.x] = input[idx + blockDim.x];

    __syncthreads();

    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            temp[index] += temp[index - stride];
        }
    }

    for (int stride = BLOCK_SIZE / 4; stride >= 1; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE) {
            temp[index + stride] += temp[index];
        }
    }

    __syncthreads();

    if (idx < size)
        output[idx] = temp[tid];
    if (idx + blockDim.x < size)
        output[idx + blockDim.x] = temp[tid + blockDim.x];
}

void checkCudaStatus(const char* message) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("CUDA Error (%s): %s\n", message, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 10;
    float hostInput[N], hostOutput[N];

    for (int i = 0; i < N; ++i)
        hostInput[i] = static_cast<float>(i + 1);

    float *devInput, *devOutput;
    cudaMalloc(&devInput, N * sizeof(float));
    cudaMalloc(&devOutput, N * sizeof(float));
    cudaMemcpy(devInput, hostInput, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaStatus("Host to Device copy");

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    scanBrentKung<<<blocksPerGrid, threadsPerBlock>>>(devInput, devOutput, N);
    checkCudaStatus("Kernel launch");

    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput, devOutput, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaStatus("Device to Host copy");

    cudaFree(devInput);
    cudaFree(devOutput);

    printf("Input Array:\n");
    for (int i = 0; i < N; ++i)
        printf("%.2f ", hostInput[i]);

    printf("\nPrefix Sum Result:\n");
    for (int i = 0; i < N; ++i)
        printf("%.2f ", hostOutput[i]);

    printf("\n");
    return 0;
}
