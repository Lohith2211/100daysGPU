#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__device__ void co_rank(const int* arr1, const int* arr2, int k, const int len1, const int len2, int* idx1_out, int* idx2_out) {
    int low = max(0, k - len2);
    int high = min(k, len1);
    
    while (low <= high) {
        int i = (low + high) / 2;
        int j = k - i;
        
        if (j < 0) {
            high = i - 1;
            continue;
        }
        if (j > len2) {
            low = i + 1;
            continue;
        }

        if (i > 0 && j < len2 && arr1[i - 1] > arr2[j]) {
            high = i - 1;
        }
        else if (j > 0 && i < len1 && arr2[j - 1] > arr1[i]) {
            low = i + 1;
        }
        else {
            *idx1_out = i;
            *idx2_out = j;
            return;
        }
    }
}

__global__ void parallel_merge(const int* arr1, const int* arr2, int* merged, const int len1, const int len2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < len1 + len2) {
        int i, j;
        co_rank(arr1, arr2, tid, len1, len2, &i, &j);
        
        if (j >= len2 || (i < len1 && arr1[i] <= arr2[j])) {
            merged[tid] = arr1[i];
        } else {
            merged[tid] = arr2[j];
        }
    }
}

int main() {
    const int len1 = 5;
    const int len2 = 5;
    int arr1[len1], arr2[len2], merged[len1 + len2];
    
    for (int i = 0; i < len1; i++) {
        arr1[i] = 2 * i;  
    }
    for (int i = 0; i < len2; i++) {
        arr2[i] = 2 * i + 1;  
    }

    printf("Array 1: ");
    for (int i = 0; i < len1; i++) {
        printf("%d ", arr1[i]);
    }
    printf("\n");

    printf("Array 2: ");
    for (int i = 0; i < len2; i++) {
        printf("%d ", arr2[i]);
    }
    printf("\n");

    int *d_arr1, *d_arr2, *d_merged;

    cudaMalloc(&d_arr1, len1 * sizeof(int));
    cudaMalloc(&d_arr2, len2 * sizeof(int));
    cudaMalloc(&d_merged, (len1 + len2) * sizeof(int));

    cudaMemcpy(d_arr1, arr1, len1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, len2 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((len1 + len2 + block.x - 1) / block.x);

    parallel_merge<<<grid, block>>>(d_arr1, d_arr2, d_merged, len1, len2);

    cudaMemcpy(merged, d_merged, (len1 + len2) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_merged);

    printf("Merged array: ");
    for (int i = 0; i < len1 + len2; i++) {
        printf("%d ", merged[i]);
    }
    printf("\n");

    return 0;
}
