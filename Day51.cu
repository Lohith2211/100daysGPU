#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define TIMESTEPS 5

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void lstm_kernel(const float* x_t, const float* h_t_prev, const float* C_t_prev,
                            const float* W, const float* U, const float* b,
                            float* h_t, float* C_t, int hidden_size, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= hidden_size * batch_size) return;

    int neuron_idx = idx % hidden_size;  
    float input_gate = sigmoid(W[neuron_idx] * x_t[idx] +
                               U[neuron_idx] * h_t_prev[idx] +
                               b[neuron_idx]);

    float forget_gate = sigmoid(W[hidden_size + neuron_idx] * x_t[idx] +
                                U[hidden_size + neuron_idx] * h_t_prev[idx] +
                                b[hidden_size + neuron_idx]);

    float output_gate = sigmoid(W[2 * hidden_size + neuron_idx] * x_t[idx] +
                                U[2 * hidden_size + neuron_idx] * h_t_prev[idx] +
                                b[2 * hidden_size + neuron_idx]);

    float candidate = tanhf(W[3 * hidden_size + neuron_idx] * x_t[idx] +
                            U[3 * hidden_size + neuron_idx] * h_t_prev[idx] +
                            b[3 * hidden_size + neuron_idx]);

    C_t[idx] = forget_gate * C_t_prev[idx] + input_gate * candidate;
    h_t[idx] = output_gate * tanhf(C_t[idx]);
}

int main() {
    int hidden_size = 128;
    int batch_size = 1;  
    int num_elements = hidden_size * batch_size;

    float *x_t, *h_t_prev, *C_t_prev, *W, *U, *b, *h_t, *C_t;
    cudaMallocManaged(&x_t, num_elements * sizeof(float));
    cudaMallocManaged(&h_t_prev, num_elements * sizeof(float));
    cudaMallocManaged(&C_t_prev, num_elements * sizeof(float));
    cudaMallocManaged(&W, 4 * hidden_size * sizeof(float));
    cudaMallocManaged(&U, 4 * hidden_size * sizeof(float));
    cudaMallocManaged(&b, 4 * hidden_size * sizeof(float));
    cudaMallocManaged(&h_t, num_elements * sizeof(float));
    cudaMallocManaged(&C_t, num_elements * sizeof(float));

    for (int i = 0; i < num_elements; i++) {
        h_t_prev[i] = 0.5f;
        C_t_prev[i] = 0.5f;
        x_t[i] = 1.0f;  
    }

    for (int i = 0; i < hidden_size; i++) {
        
        W[i] = 0.5f;
        U[i] = 0.5f;
        b[i] = 0.1f;

        W[hidden_size + i] = 0.5f;
        U[hidden_size + i] = 0.5f;
        b[hidden_size + i] = 0.2f;

        W[2 * hidden_size + i] = 0.5f;
        U[2 * hidden_size + i] = 0.5f;
        b[2 * hidden_size + i] = 0.3f;

        W[3 * hidden_size + i] = 0.5f;
        U[3 * hidden_size + i] = 0.5f;
        b[3 * hidden_size + i] = 0.0f;
    }

    int threads_per_block = 128;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "LSTM processing over " << TIMESTEPS << " timesteps:" << std::endl;

    for (int t = 0; t < TIMESTEPS; t++) {
        lstm_kernel<<<num_blocks, threads_per_block>>>(x_t, h_t_prev, C_t_prev,
                                                         W, U, b, h_t, C_t,
                                                         hidden_size, batch_size);
        cudaDeviceSynchronize();

        for (int i = 0; i < num_elements; i++) {
            h_t_prev[i] = h_t[i];
            C_t_prev[i] = C_t[i];
        }

        std::cout << "After timestep " << t + 1 << ": h_t[0] = " << h_t[0]
                  << ", C_t[0] = " << C_t[0] << std::endl;
    }

    cudaFree(x_t);
    cudaFree(h_t_prev);
    cudaFree(C_t_prev);
    cudaFree(W);
    cudaFree(U);
    cudaFree(b);
    cudaFree(h_t);
    cudaFree(C_t);

    return 0;
}