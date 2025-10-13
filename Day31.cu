#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

__global__ void gameOfLifeKernel(const int *in, int *out, int width, int height) {
    
    extern __shared__ int sTile[];

    int bx = blockDim.x;
    int by = blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int globalX = blockIdx.x * bx + tx;
    int globalY = blockIdx.y * by + ty;
    
    int sWidth = bx + 2;
    int sX = tx + 1;
    int sY = ty + 1;
    
    if (globalX < width && globalY < height)
        sTile[sY * sWidth + sX] = in[globalY * width + globalX];
    else
        sTile[sY * sWidth + sX] = 0;
    
   
    if (ty == 0) {
        int gY = globalY - 1;
        int sY_top = 0;
        if (gY >= 0 && globalX < width)
            sTile[sY_top * sWidth + sX] = in[gY * width + globalX];
        else
            sTile[sY_top * sWidth + sX] = 0;
    }
   
    if (ty == by - 1) {
        int gY = globalY + 1;
        int sY_bottom = by + 1;
        if (gY < height && globalX < width)
            sTile[sY_bottom * sWidth + sX] = in[gY * width + globalX];
        else
            sTile[sY_bottom * sWidth + sX] = 0;
    }
   
    if (tx == 0) {
        int gX = globalX - 1;
        int sX_left = 0;
        if (gX >= 0 && globalY < height)
            sTile[sY * sWidth + sX_left] = in[globalY * width + gX];
        else
            sTile[sY * sWidth + sX_left] = 0;
    }
    
    if (tx == bx - 1) {
        int gX = globalX + 1;
        int sX_right = bx + 1;
        if (gX < width && globalY < height)
            sTile[sY * sWidth + sX_right] = in[globalY * width + gX];
        else
            sTile[sY * sWidth + sX_right] = 0;
    }
 
    if (tx == 0 && ty == 0) {
        int gX = globalX - 1;
        int gY = globalY - 1;
        int sIndex = 0; 
        if (gX >= 0 && gY >= 0)
            sTile[sIndex] = in[gY * width + gX];
        else
            sTile[sIndex] = 0;
    }
    
    if (tx == bx - 1 && ty == 0) {
        int gX = globalX + 1;
        int gY = globalY - 1;
        int sIndex = (0 * sWidth) + (bx + 1);
        if (gX < width && gY >= 0)
            sTile[sIndex] = in[gY * width + gX];
        else
            sTile[sIndex] = 0;
    }
    
    if (tx == 0 && ty == by - 1) {
        int gX = globalX - 1;
        int gY = globalY + 1;
        int sIndex = (by + 1) * sWidth + 0;
        if (gX >= 0 && gY < height)
            sTile[sIndex] = in[gY * width + gX];
        else
            sTile[sIndex] = 0;
    }
    
    if (tx == bx - 1 && ty == by - 1) {
        int gX = globalX + 1;
        int gY = globalY + 1;
        int sIndex = (by + 1) * sWidth + (bx + 1);
        if (gX < width && gY < height)
            sTile[sIndex] = in[gY * width + gX];
        else
            sTile[sIndex] = 0;
    }
    
    
    __syncthreads();
    
   
    if (globalX < width && globalY < height) {
       
        int sum = 0;
        sum += sTile[(sY - 1) * sWidth + (sX - 1)];
        sum += sTile[(sY - 1) * sWidth + (sX)];
        sum += sTile[(sY - 1) * sWidth + (sX + 1)];
        sum += sTile[(sY) * sWidth + (sX - 1)];
        sum += sTile[(sY) * sWidth + (sX + 1)];
        sum += sTile[(sY + 1) * sWidth + (sX - 1)];
        sum += sTile[(sY + 1) * sWidth + (sX)];
        sum += sTile[(sY + 1) * sWidth + (sX + 1)];
        
        int cell = sTile[sY * sWidth + sX];
        int newState = 0;
        
        if (cell == 1 && (sum == 2 || sum == 3))
            newState = 1;
        else if (cell == 0 && sum == 3)
            newState = 1;
        else
            newState = 0;
        
        out[globalY * width + globalX] = newState;
    }
}


int main() {
    
    const int width = 64;
    const int height = 64;
    const int size = width * height;
    
    int *h_grid   = (int*)malloc(size * sizeof(int));
    int *h_result = (int*)malloc(size * sizeof(int));
    
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        h_grid[i] = rand() % 2; 
    }
    
    int *d_grid, *d_result;
    cudaMalloc(&d_grid, size * sizeof(int));
    cudaMalloc(&d_result, size * sizeof(int));
    
    cudaMemcpy(d_grid, h_grid, size * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    size_t sharedSize = (block.x + 2) * (block.y + 2) * sizeof(int);
    
    gameOfLifeKernel<<<grid, block, sharedSize>>>(d_grid, d_result, width, height);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Game of Life Grid After One Iteration:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%d ", h_result[y * width + x]);
        }
        printf("\n");
    }
    
    cudaFree(d_grid);
    cudaFree(d_result);
    free(h_grid);
    free(h_result);
    
    return 0;
}