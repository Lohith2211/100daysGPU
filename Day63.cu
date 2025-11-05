#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    if(err != cudaSuccess) {\
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);\
    }

__global__ void groupNormForward(const float *x, float *y, 
                                 const float *gamma, const float *beta,
                                 int N, int C, int H, int W, int G, float eps) {
   
    int n = blockIdx.x;
    int g = blockIdx.y;
    
    int group_channels = C / G;
    int spatial_size = H * W;
    int group_size = group_channels * spatial_size;
    int start_c = g * group_channels;
    
    int tid = threadIdx.x;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c_offset = i / spatial_size;  
        int s = i % spatial_size;           
        int channel = start_c + c_offset;
        int idx = n * C * spatial_size + channel * spatial_size + s;
        float val = x[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    extern __shared__ float shared[];
    float* s_sum = shared;              
    float* s_sum_sq = &shared[blockDim.x]; 

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = s_sum[0] / group_size;
    float var = s_sum_sq[0] / group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);
    __syncthreads(); 
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c_offset = i / spatial_size;
        int s = i % spatial_size;
        int channel = start_c + c_offset;
        int idx = n * C * spatial_size + channel * spatial_size + s;
        float val = x[idx];
        float norm = (val - mean) * inv_std;
        y[idx] = gamma[channel] * norm + beta[channel];
    }
}

int main() {
    
    int N = 1;   
    int C = 4;   
    int H = 2;   
    int W = 2;   
    int G = 2;   
    float eps = 1e-5;
    int tensor_size = N * C * H * W;
    
    float h_x[tensor_size];
    float h_y[tensor_size];
    float h_gamma[C];
    float h_beta[C];
    
    for (int i = 0; i < tensor_size; i++) {
        h_x[i] = static_cast<float>(i);
    }
    for (int i = 0; i < C; i++) {
        h_gamma[i] = 1.0f; 
        h_beta[i] = 0.0f;   
    }
    
    float *d_x, *d_y, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_x, tensor_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, tensor_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, C * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_x, h_x, tensor_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice));
    
    int group_channels = C / G;
    int spatial_size = H * W;
    int group_size = group_channels * spatial_size;
    int threadsPerBlock = (group_size < 128) ? group_size : 128;
    size_t sharedMemSize = threadsPerBlock * 2 * sizeof(float);
    dim3 grid(N, G);
    
    groupNormForward<<<grid, threadsPerBlock, sharedMemSize>>>(d_x, d_y, d_gamma, d_beta,
                                                                N, C, H, W, G, eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_y, d_y, tensor_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Group Norm Forward Output:" << std::endl;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = n * C * H * W + c * H * W + h * W + w;
                    std::cout << h_y[idx] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    
    return 0;
}