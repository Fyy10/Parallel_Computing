#include "cuda.h"
#include "stdio.h"

void vecAdd(float *h_A, float *h_B, float *h_C, int n);
__global__ void AddKernel(float *d_A, float *d_B, float *d_C, int n);

int main() {
    const int n = 10;
    float a[n], b[n], c[n];
    // generate float data
    for (int i = 0; i < n; i++) {
        a[i] = float(i);
        b[i] = float(i + 1);
    }

    printf("Array a:\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", a[i]);
    }
    printf("\n");

    printf("Array b:\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", b[i]);
    }
    printf("\n");

    // vector addition
    vecAdd(a, b, c, n);

    printf("Array c (a + b):\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");

    return 0;
}

void vecAdd(float *h_A, float *h_B, float *h_C, int n) {
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaError_t err;
    // copy A
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // copy B
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // copy C
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    // define the blocks in a grid
    dim3 DimGrid(1);
    // define the threads in a block
    dim3 DimBlock(n);

    AddKernel<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void AddKernel(float *d_A, float *d_B, float *d_C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d blockIdx: %d %d %d\n", i, blockIdx.x, blockIdx.y, blockIdx.z);
    printf("%d blockDim: %d %d %d\n", i, blockDim.x, blockDim.y, blockDim.z);
    printf("%d threadIdx: %d %d %d\n", i, threadIdx.x, threadIdx.y, threadIdx.z);
    if (i < n) d_C[i] = d_A[i] + d_B[i];
}
