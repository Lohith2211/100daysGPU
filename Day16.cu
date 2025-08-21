#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NaiveBayesKernel.cuh"
#include "NaiveBayesTrain.cuh"

#define SHARED_SIZE 20

__global__ void computePriorsAndLikelihood(
    int* d_Dataset, int* d_priors, int* d_likelihoods,
    int numSamples, int numFeatures, int numClasses, int numFeatureValues
) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int local_d_priors[SHARED_SIZE];
    __shared__ int local_d_likelihoods[SHARED_SIZE];

    if (threadId < numSamples) {

        int classLabel = d_Dataset[threadId * (numFeatures + 1) + numFeatures]; 

        atomicAdd(&local_d_priors[classLabel], 1);

        for (int fIdx = 0; fIdx < numFeatures; ++fIdx) {
            int featureValue = d_Dataset[threadId * (numFeatures + 1) + fIdx];
            int likelihoodIndex = classLabel * numFeatures * numFeatureValues + (fIdx * numFeatureValues) + featureValue;

            atomicAdd(&local_d_likelihoods[likelihoodIndex], 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int c = 0; c < numClasses; ++c) {
            atomicAdd(&d_priors[c], local_d_priors[c]);
        }

        for (int l = 0; l < numClasses * numFeatures * numFeatureValues; ++l) {
            atomicAdd(&d_likelihoods[l], local_d_likelihoods[l]);
        }
    }
}