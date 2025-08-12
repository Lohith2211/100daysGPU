#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void hybridSparseKernel(
    const float* mat, const float* vec,
    float* ell_data, int* ell_indices,
    float* coo_data, int* coo_rows, int* coo_cols,
    float* result, const int ell_limit,
    const int numRows, const int numCols,
    int* coo_counter_global
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    int nz_in_row = 0;

    for (int col = 0; col < numCols; ++col) {
        float val = mat[row * numCols + col];
        if (val != 0) {
            if (nz_in_row < ell_limit) {

                ell_data[nz_in_row * numRows + row] = val;
                ell_indices[nz_in_row * numRows + row] = col;
            } else {

                int idx = atomicAdd(coo_counter_global, 1);
                coo_data[idx] = val;
                coo_rows[idx] = row;
                coo_cols[idx] = col;
            }
            nz_in_row++;
        }
    }

    // Padding ELL for unused slots
    for (int i = nz_in_row; i < ell_limit; ++i) {
        ell_data[i * numRows + row] = 0.0f;
        ell_indices[i * numRows + row] = -1;
    }

    float sum = 0.0f;

    for (int i = 0; i < ell_limit; ++i) {
        int colIdx = ell_indices[i * numRows + row];
        if (colIdx != -1)
            sum += ell_data[i * numRows + row] * vec[colIdx];
    }

    int total_coo = *coo_counter_global;
    for (int i = 0; i < total_coo; ++i) {
        if (coo_rows[i] == row)
            sum += coo_data[i] * vec[coo_cols[i]];
    }

    result[row] = sum;
}

int main() {
    const int rows = 1000;
    const int cols = 1000;
    const int ell_threshold = 20;

    float* h_matrix = new float[rows * cols];
    float* h_vector = new float[cols];
    float* h_output = new float[rows];

    float* h_ell_data = new float[rows * ell_threshold]();
    int* h_ell_indices = new int[rows * ell_threshold]();
    float* h_coo_data = new float[rows * cols]();
    int* h_coo_rows = new int[rows * cols]();
    int* h_coo_cols = new int[rows * cols]();

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            h_matrix[i * cols + j] = ((i + j) % 3 == 0) ? (i + j) : 0.0f;

    for (int i = 0; i < cols; ++i)
        h_vector[i] = 1.0f;

    float *d_matrix, *d_vector, *d_result;
    float *d_ell_data, *d_coo_data;
    int *d_ell_indices, *d_coo_rows, *d_coo_cols;
    int* d_coo_counter;

    CHECK_CUDA(cudaMalloc(&d_matrix, sizeof(float) * rows * cols));
    CHECK_CUDA(cudaMalloc(&d_vector, sizeof(float) * cols));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float) * rows));

    CHECK_CUDA(cudaMalloc(&d_ell_data, sizeof(float) * rows * ell_threshold));
    CHECK_CUDA(cudaMalloc(&d_ell_indices, sizeof(int) * rows * ell_threshold));
    CHECK_CUDA(cudaMalloc(&d_coo_data, sizeof(float) * rows * cols));
    CHECK_CUDA(cudaMalloc(&d_coo_rows, sizeof(int) * rows * cols));
    CHECK_CUDA(cudaMalloc(&d_coo_cols, sizeof(int) * rows * cols));
    CHECK_CUDA(cudaMalloc(&d_coo_counter, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_coo_counter, 0, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_matrix, h_matrix, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vector, h_vector, sizeof(float) * cols, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    hybridSparseKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_matrix, d_vector,
        d_ell_data, d_ell_indices,
        d_coo_data, d_coo_rows, d_coo_cols,
        d_result, ell_threshold,
        rows, cols,
        d_coo_counter
    );

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    std::cout << "Kernel execution time: " << elapsed / 1000.0f << " sec" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_output, d_result, sizeof(float) * rows, cudaMemcpyDeviceToHost));

    int h_coo_count;
    CHECK_CUDA(cudaMemcpy(&h_coo_count, d_coo_counter, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Total COO elements: " << h_coo_count << std::endl;

    FILE* fout = fopen("cuda_results.txt", "w");
    if (fout) {
        for (int i = 0; i < rows; ++i)
            fprintf(fout, "%.10f\n", h_output[i]);
        fclose(fout);
        std::cout << "Results written to cuda_results.txt" << std::endl;
    }

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
    cudaFree(d_ell_data);
    cudaFree(d_ell_indices);
    cudaFree(d_coo_data);
    cudaFree(d_coo_rows);
    cudaFree(d_coo_cols);
    cudaFree(d_coo_counter);

    delete[] h_matrix;
    delete[] h_vector;
    delete[] h_output;
    delete[] h_ell_data;
    delete[] h_ell_indices;
    delete[] h_coo_data;
    delete[] h_coo_rows;
    delete[] h_coo_cols;

    return 0;
}
