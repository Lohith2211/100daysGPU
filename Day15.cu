#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <limits>

#include "helper.cuh"
#include "kernels.cuh"

#define BLOCK_SIZE 1024
#define THREADS_PER_BLOCK 1024
#define NEGATIVE_INFINITY -1e38f

void flashAttention2BackwardPass(
    const float* query, const float* key, const float* value,
    const float* output, const float* dOutput,
    float* dQuery, float* dKey, float* dValue,
    int numTokens, int dim, int blockCols, int blockRows,
    float* hostL
) {
    float scale = 1.0f / sqrtf((float)dim);

    float* deviceD;
    cudaMalloc((void**)&deviceD, numTokens * sizeof(float));
    computeDKernel<<<(numTokens + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOutput, output, deviceD, numTokens, dim);
    cudaDeviceSynchronize();

    float* hostD = (float*)malloc(numTokens * sizeof(float));
    cudaMemcpy(hostD, deviceD, numTokens * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(dQuery, 0, numTokens * dim * sizeof(float));
    cudaMemset(dKey, 0, numTokens * dim * sizeof(float));
    cudaMemset(dValue, 0, numTokens * dim * sizeof(float));

    for (int colBlockIdx = 0; colBlockIdx < (numTokens + blockCols - 1) / blockCols; ++colBlockIdx) {
        float* hostKeyBlock = (float*)malloc(blockCols * dim * sizeof(float));
        float* hostValueBlock = (float*)malloc(blockCols * dim * sizeof(float));

        cudaMemcpy(hostKeyBlock, key + colBlockIdx * blockCols * dim, blockCols * dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostValueBlock, value + colBlockIdx * blockCols * dim, blockCols * dim * sizeof(float), cudaMemcpyDeviceToHost);

        float *deviceKeyBlock, *deviceValueBlock;
        cudaMalloc((void**)&deviceKeyBlock, blockCols * dim * sizeof(float));
        cudaMalloc((void**)&deviceValueBlock, blockCols * dim * sizeof(float));
        cudaMemcpy(deviceKeyBlock, hostKeyBlock, blockCols * dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceValueBlock, hostValueBlock, blockCols * dim * sizeof(float), cudaMemcpyHostToDevice);

        float *tempGradKeyBlock, *tempGradValueBlock;
        cudaMalloc((void**)&tempGradKeyBlock, blockCols * dim * sizeof(float));
        cudaMalloc((void**)&tempGradValueBlock, blockCols * dim * sizeof(float));
        cudaMemset(tempGradKeyBlock, 0, blockCols * dim * sizeof(float));
        cudaMemset(tempGradValueBlock, 0, blockCols * dim * sizeof(float));

        for (int rowBlockIdx = 0; rowBlockIdx < (numTokens + blockRows - 1) / blockRows; ++rowBlockIdx) {
            const float* queryBlock = query + rowBlockIdx * blockRows * dim;
            const float* dOutputBlock = dOutput + rowBlockIdx * blockRows * dim;
            float* tempGradQueryBlock;
            cudaMalloc((void**)&tempGradQueryBlock, blockRows * dim * sizeof(float));
            cudaMemset(tempGradQueryBlock, 0, blockRows * dim * sizeof(float));

            const float* LBlock = hostL + rowBlockIdx * blockRows;
            const float* DBlock = hostD + rowBlockIdx * blockRows;

            float *deviceSBlock, *devicePBlock, *deviceDPBlock, *deviceDSBlock;
            cudaMalloc((void**)&deviceSBlock, blockRows * blockCols * sizeof(float));
            cudaMalloc((void**)&devicePBlock, blockRows * blockCols * sizeof(float));
            cudaMalloc((void**)&deviceDPBlock, blockRows * blockCols * sizeof(float));
            cudaMalloc((void**)&deviceDSBlock, blockRows * blockCols * sizeof(float));

            float* deviceMaxSBlock;
            cudaMalloc((void**)&deviceMaxSBlock, blockRows * sizeof(float));

            computeSiKernel<<<(blockRows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                queryBlock, deviceKeyBlock, deviceSBlock, blockRows, blockCols, dim, scale);
            cudaDeviceSynchronize();

            findRowMaxSiKernel<<<(blockRows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                deviceSBlock, deviceMaxSBlock, blockRows, blockCols);
            cudaDeviceSynchronize();

            computePiKernel<<<(blockRows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                deviceSBlock, LBlock, devicePBlock, blockRows, blockCols, deviceMaxSBlock);
            cudaDeviceSynchronize();

            computeDViKernel<<<(dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                devicePBlock, dOutputBlock, tempGradValueBlock, blockRows, blockCols, dim);
            cudaDeviceSynchronize();

            computeDPiKernel<<<(blockRows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                dOutputBlock, deviceValueBlock, deviceDPBlock, blockRows, blockCols, dim);
            cudaDeviceSynchronize();

            computeDSiKernel<<<(blockRows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                devicePBlock, deviceDPBlock, DBlock, deviceDSBlock, blockRows, blockCols);
            cudaDeviceSynchronize();

            computeDQiKernel<<<(blockRows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                deviceDSBlock, deviceKeyBlock, tempGradQueryBlock, blockRows, dim, blockCols);
            cudaDeviceSynchronize();

            computeDKjKernel<<<(blockCols + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                deviceDSBlock, queryBlock, tempGradKeyBlock, blockCols, dim, blockRows);
            cudaDeviceSynchronize();

            accumulateDQKernel<<<(blockRows * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                dQuery, tempGradQueryBlock, blockRows, dim, rowBlockIdx * blockRows * dim);
            cudaDeviceSynchronize();

            cudaFree(deviceSBlock);
            cudaFree(devicePBlock);
            cudaFree(deviceDPBlock);
            cudaFree(deviceDSBlock);
            cudaFree(deviceMaxSBlock);
            cudaFree(tempGradQueryBlock);
        }

        accumulateDKVjKernel<<<(blockCols * dim + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            dKey, dValue, tempGradKeyBlock, tempGradValueBlock, blockCols, dim, colBlockIdx * blockCols * dim);
        cudaDeviceSynchronize();

        cudaFree(deviceKeyBlock);
        cudaFree(deviceValueBlock);
        cudaFree(tempGradKeyBlock);
        cudaFree(tempGradValueBlock);
        free(hostKeyBlock);
        free(hostValueBlock);
    }

    cudaFree(deviceD);
    free(hostD);
}
