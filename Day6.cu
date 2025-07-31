#include <cuda_runtime.h>
#include <iostream>

constexpr int MATRIX_WIDTH = 1024;
constexpr int MATRIX_HEIGHT = 1024;

__global__ void transposeKernel(const float* src, float* dst, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int srcIdx = row * width + col;
        int dstIdx = col * height + row;
        dst[dstIdx] = src[srcIdx];
    }
}

void verifyCudaCall(const char* context) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << context << ": CUDA error: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int width = MATRIX_WIDTH;
    int height = MATRIX_HEIGHT;
    size_t totalBytes = width * height * sizeof(float);

    float* hostSrc = (float*)malloc(totalBytes);
    float* hostDst = (float*)malloc(totalBytes);
    for (int i = 0; i < width * height; ++i) {
        hostSrc[i] = static_cast<float>(i);
    }

    float *deviceSrc, *deviceDst;
    cudaMalloc(&deviceSrc, totalBytes);
    cudaMalloc(&deviceDst, totalBytes);

    cudaMemcpy(deviceSrc, hostSrc, totalBytes, cudaMemcpyHostToDevice);
    verifyCudaCall("Memcpy host to device");

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceSrc, deviceDst, width, height);
    cudaDeviceSynchronize();
    verifyCudaCall("Kernel execution");

    cudaMemcpy(hostDst, deviceDst, totalBytes, cudaMemcpyDeviceToHost);
    verifyCudaCall("Memcpy device to host");

    bool isCorrect = true;
    for (int col = 0; col < width && isCorrect; ++col) {
        for (int row = 0; row < height; ++row) {
            if (hostDst[col * height + row] != hostSrc[row * width + col]) {
                isCorrect = false;
                break;
            }
        }
    }

    std::cout << (isCorrect ? "Transpose successful!" : "Transpose verification failed!") << std::endl;

    cudaFree(deviceSrc);
    cudaFree(deviceDst);
    free(hostSrc);
    free(hostDst);

    return 0;
}
