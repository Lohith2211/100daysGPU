#include <hip/hip_runtime.h>
#include <iostream>

#define N 100              
#define T 1000             
#define DX 0.1             
#define DT 0.01            
#define ALPHA 0.1         
#define BLOCK_SIZE 16      
#define TILE_SIZE (BLOCK_SIZE - 2)  

#define CHECK_HIP_ERROR(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

__global__ void heat_diffusion_optimized(const double* __restrict__ u, 
                                       double* __restrict__ u_new, 
                                       const int n) {
    __shared__ double tile[BLOCK_SIZE][BLOCK_SIZE];
    
    int gx = blockIdx.x * TILE_SIZE + threadIdx.x - 1;
    int gy = blockIdx.y * TILE_SIZE + threadIdx.y - 1;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (gx >= 0 && gx < n && gy >= 0 && gy < n) {
        tile[ty][tx] = u[gx * n + gy];
    } else {
        tile[ty][tx] = 0.0;
    }
    
    __syncthreads();

    if (tx > 0 && tx < BLOCK_SIZE-1 && ty > 0 && ty < BLOCK_SIZE-1) {
        gx = blockIdx.x * TILE_SIZE + threadIdx.x - 1;
        gy = blockIdx.y * TILE_SIZE + threadIdx.y - 1;
        
        if (gx > 0 && gx < n-1 && gy > 0 && gy < n-1) {
            double d2u_dx2 = (tile[ty][tx+1] - 2.0*tile[ty][tx] + tile[ty][tx-1]) / (DX * DX);
            double d2u_dy2 = (tile[ty+1][tx] - 2.0*tile[ty][tx] + tile[ty-1][tx]) / (DX * DX);
            
            u_new[gx * n + gy] = tile[ty][tx] + ALPHA * DT * (d2u_dx2 + d2u_dy2);
        }
    }
}

int main() {
    double *u, *u_new;
    double *d_u, *d_u_new;

    CHECK_HIP_ERROR(hipHostMalloc(&u, N * N * sizeof(double)));
    CHECK_HIP_ERROR(hipHostMalloc(&u_new, N * N * sizeof(double)));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i * N + j] = (i == 0 || j == 0 || i == N-1 || j == N-1) ? 100.0 : 0.0;
            u_new[i * N + j] = u[i * N + j];
        }
    }

    CHECK_HIP_ERROR(hipMalloc(&d_u, N * N * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&d_u_new, N * N * sizeof(double)));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_u, u, N * N * sizeof(double), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_u_new, u_new, N * N * sizeof(double), hipMemcpyHostToDevice, stream));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    for (int t = 0; t < T; t++) {
        heat_diffusion_optimized<<<gridSize, blockSize, 0, stream>>>(d_u, d_u_new, N);
        
        double *temp = d_u;
        d_u = d_u_new;
        d_u_new = temp;
    }

    CHECK_HIP_ERROR(hipMemcpyAsync(u, d_u, N * N * sizeof(double), hipMemcpyDeviceToHost, stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << u[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipFree(d_u));
    CHECK_HIP_ERROR(hipFree(d_u_new));
    CHECK_HIP_ERROR(hipHostFree(u));
    CHECK_HIP_ERROR(hipHostFree(u_new));

    return 0;
} 