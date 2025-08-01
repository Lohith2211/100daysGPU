#include <iostream>
#include <cuda_runtime.h>

#define FILTER_SIZE 5

__constant__ float filter[FILTER_SIZE];

__global__ void convolution1D(const float* input, float* output, int length) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < length) {
        float sum = 0.0f;
        int halfFilter = FILTER_SIZE / 2;

        for (int offset = -halfFilter; offset <= halfFilter; ++offset) {
            int neighborIdx = idx + offset;
            if (neighborIdx >= 0 && neighborIdx < length) {
                sum += input[neighborIdx] * filter[offset + halfFilter];
            }
        }
        output[idx] = sum;
    }
}

void checkCudaErrors(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << ": CUDA Error: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 10;

    float h_input[N];
    float h_output[N];
    float h_filter[FILTER_SIZE];

    for (int i = 0; i < FILTER_SIZE; ++i) {
        h_filter[i] = static_cast<float>(i);
    }

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors("Failed copying input to device");

    cudaMemcpyToSymbol(filter, h_filter, FILTER_SIZE * sizeof(float));
    checkCudaErrors("Failed copying filter to constant memory");

    dim3 blockDim(32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    convolution1D<<<gridDim, blockDim>>>(d_input, d_output, N);

    cudaDeviceSynchronize();
    checkCudaErrors("Kernel execution failed");

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors("Failed copying output to host");

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Input Array:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << "\nFilter:\n";
    for (int i = 0; i < FILTER_SIZE; ++i) {
        std::cout << h_filter[i] << " ";
    }
    std::cout << "\nOutput Array:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
