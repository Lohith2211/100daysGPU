#include "solve.h"
#include <cuda_runtime.h>

__global__
void flockKernel(const float* agents, float* agents_next, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    int base = 4 * i;
    float x  = agents[base + 0];
    float y  = agents[base + 1];
    float vx = agents[base + 2];
    float vy = agents[base + 3];
    
    const float r      = 5.0f;       
    const float r_sq   = r * r;     
    const float alpha  = 0.05f;      

    float sum_vx = 0.0f;
    float sum_vy = 0.0f;
    int neighborCount = 0;
    
    for (int j = 0; j < N; j++)
    {
        if (j == i) continue;  

        int jbase = 4 * j;
        float xj  = agents[jbase + 0];
        float yj  = agents[jbase + 1];
        
        float dx = xj - x;
        float dy = yj - y;
        float dist_sq = dx*dx + dy*dy;
        
        if (dist_sq < r_sq) {
            
            sum_vx += agents[jbase + 2];
            sum_vy += agents[jbase + 3];
            neighborCount++;
        }
    }
    
    float new_vx = vx;
    float new_vy = vy;
    if (neighborCount > 0)
    {
        float avg_vx = sum_vx / neighborCount;
        float avg_vy = sum_vy / neighborCount;
       
        new_vx = vx + alpha * (avg_vx - vx);
        new_vy = vy + alpha * (avg_vy - vy);
    }
    
    float new_x = x + new_vx;
    float new_y = y + new_vy;
    
    agents_next[base + 0] = new_x;
    agents_next[base + 1] = new_y;
    agents_next[base + 2] = new_vx;
    agents_next[base + 3] = new_vy;
}

void solve(const float* agents, float* agents_next, int N)
{
   
    float *d_agents     = nullptr;
    float *d_agentsNext = nullptr;
    
    size_t size = 4 * N * sizeof(float);
    
    cudaMalloc((void**)&d_agents,     size);
    cudaMalloc((void**)&d_agentsNext, size);
    
    cudaMemcpy(d_agents, agents, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    
    flockKernel<<<gridSize, blockSize>>>(d_agents, d_agentsNext, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(agents_next, d_agentsNext, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_agents);
    cudaFree(d_agentsNext);
}