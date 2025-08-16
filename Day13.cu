#include "helper_functions.h"
#include "cuda_kernels.h"

int main() {

    const int BATCH_SIZE = 1;
    const int NUM_INPUT_NEURONS = 256;
    const int NUM_OUTPUT_NEURONS = 256;

    const size_t NUM_INPUT_ELEMENTS = BATCH_SIZE * NUM_INPUT_NEURONS;
    const size_t NUM_WEIGHT_ELEMENTS = NUM_INPUT_NEURONS * NUM_OUTPUT_NEURONS;
    const size_t NUM_BIAS_ELEMENTS = NUM_OUTPUT_NEURONS;
    const size_t NUM_OUTPUT_ELEMENTS = BATCH_SIZE * NUM_OUTPUT_NEURONS;

    float* h_input = new float[NUM_INPUT_ELEMENTS];
    float* h_weights = new float[NUM_WEIGHT_ELEMENTS];
    float* h_bias = new float[NUM_BIAS_ELEMENTS];
    float* h_output = new float[NUM_OUTPUT_ELEMENTS];

    initializeRandomMatrix(h_input, NUM_INPUT_ELEMENTS, -1.0f, 1.0f);
    initializeRandomMatrix(h_weights, NUM_WEIGHT_ELEMENTS, -1.0f, 1.0f);
    initializeRandomMatrix(h_bias, NUM_BIAS_ELEMENTS, -0.1f, 0.1f);

    float *d_input, *d_weights, *d_bias, *d_output;
    allocateDeviceMemory(&d_input, NUM_INPUT_ELEMENTS);
    allocateDeviceMemory(&d_weights, NUM_WEIGHT_ELEMENTS);
    allocateDeviceMemory(&d_bias, NUM_BIAS_ELEMENTS);
    allocateDeviceMemory(&d_output, NUM_OUTPUT_ELEMENTS);

    cudaMemcpy(d_input, h_input, NUM_INPUT_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, NUM_WEIGHT_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, NUM_BIAS_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    performLinearLayerOperation(
        cublasHandle,
        d_input,
        d_weights,
        d_bias,
        d_output,
        BATCH_SIZE,
        NUM_INPUT_NEURONS,
        NUM_OUTPUT_NEURONS
    );

    cudaMemcpy(h_output, d_output, NUM_OUTPUT_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(cublasHandle);
    freeDeviceMemory(d_input);
    freeDeviceMemory(d_weights);
    freeDeviceMemory(d_bias);
    freeDeviceMemory(d_output);
    delete[] h_input;
    delete[] h_weights;
    delete[] h_bias;
    delete[] h_output;

    return 0;
}
