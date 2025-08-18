#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
#define CHUNK_SIZE 256

using namespace cv;
using namespace std;

__constant__ float kx_const[CHUNK_SIZE], ky_const[CHUNK_SIZE], kz_const[CHUNK_SIZE];

__global__ void processKernel(float* realOut, float* imagOut, float* magnitude,
                              float* posX, float* posY, float* intensity,
                              float* realMu, float* imagMu, int M) {
    int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    
    float x = posX[idx]; 
    float y = posY[idx]; 
    float z = intensity[idx];

    float realVal = realOut[idx]; 
    float imagVal = imagOut[idx];

    for (int i = 0; i < M; i++) {
        float angle = 2 * PI * (kx_const[i] * x + ky_const[i] * y + kz_const[i] * z);
        float cosVal = __cosf(angle);
        float sinVal = __sinf(angle);

        realVal += realMu[i] * cosVal - imagMu[i] * sinVal;
        imagVal += imagMu[i] * cosVal + realMu[i] * sinVal;
    }

    realOut[idx] = realVal;
    imagOut[idx] = imagVal;
}

int main() {
    Mat img = imread("lena_gray.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not open the image!" << endl;
        return -1;
    }

    img.convertTo(img, CV_32F, 1.0 / 255);

    int totalPixels = img.rows * img.cols;
    int numFreq = 256;

    float *posX, *posY, *intensity;
    float *realMu, *imagMu;
    float *realOut, *imagOut, *magnitude;

    cudaMallocManaged(&posX, totalPixels * sizeof(float));
    cudaMallocManaged(&posY, totalPixels * sizeof(float));
    cudaMallocManaged(&intensity, totalPixels * sizeof(float));
    cudaMallocManaged(&realMu, numFreq * sizeof(float));
    cudaMallocManaged(&imagMu, numFreq * sizeof(float));
    cudaMallocManaged(&realOut, totalPixels * sizeof(float));
    cudaMallocManaged(&imagOut, totalPixels * sizeof(float));
    cudaMallocManaged(&magnitude, totalPixels * sizeof(float));

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int idx = i * img.cols + j;
            posX[idx] = (float)j / img.cols;
            posY[idx] = (float)i / img.rows;
            intensity[idx] = img.at<float>(i, j);
            realOut[idx] = intensity[idx];
            imagOut[idx] = 0.0f;
        }
    }

    for (int i = 0; i < numFreq; i++) {
        realMu[i] = static_cast<float>(rand()) / RAND_MAX;
        imagMu[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < numFreq / CHUNK_SIZE; i++) {
        cudaMemcpyToSymbol(kx_const, &posX[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(ky_const, &posY[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(kz_const, &intensity[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));

        processKernel<<<totalPixels / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            realOut, imagOut, magnitude, posX, posY, intensity, realMu, imagMu, CHUNK_SIZE
        );
        cudaDeviceSynchronize();
    }

    Mat output(img.rows, img.cols, CV_32F);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int idx = i * img.cols + j;
            output.at<float>(i, j) = sqrt(realOut[idx] * realOut[idx] + imagOut[idx] * imagOut[idx]);
        }
    }

    normalize(output, output, 0, 255, NORM_MINMAX);
    output.convertTo(output, CV_8U);
    imwrite("output.jpg", output);

    cudaFree(posX);
    cudaFree(posY);
    cudaFree(intensity);
    cudaFree(realMu);
    cudaFree(imagMu);
    cudaFree(realOut);
    cudaFree(imagOut);
    cudaFree(magnitude);

    cout << "Processed image saved as output.jpg" << endl;
    return 0;
}
