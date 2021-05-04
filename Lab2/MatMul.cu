#include "cuda.h"
#include "stdio.h"

void MatMul(float *A, float *B, float *C, int m, int n, int p);
__global__ void MatMulKernel(float *A, float *B, float *C, int m, int n, int p);
void printMat(float *A, int m, int n);

int main() {
    const int m = 10, n = 5, p = 10;
    float a[m][n], b[n][p], c[m][p];
    // generate A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = float(i + j);
        }
    }
    // generate B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            b[i][j] = float(i + j);
        }
    }
    printf("Mat A:\n");
    printMat(*a, m, n);
    printf("Mat B:\n");
    printMat(*b, n, p);

    MatMul(*a, *b, *c, m, n, p);

    printf("Mat C:\n");
    printMat(*c, m, p);
    return 0;
}

void printMat(float *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("shape: (%d, %d)\n", m, n);
}

void MatMul(float *A, float *B, float *C, int m, int n, int p) {
    int sizeA = m * n * sizeof(float);
    int sizeB = n * p * sizeof(float);
    int sizeC = m * p * sizeof(float);
    float *dA, *dB, *dC;

    cudaMalloc((void **)&dA, sizeA);
    cudaMalloc((void **)&dB, sizeB);
    cudaMalloc((void **)&dC, sizeC);
    cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeC, cudaMemcpyHostToDevice);

    // 1 block
    dim3 DimGrid(1);
    // m * n threads
    dim3 DimBlock(m, p);

    MatMulKernel<<<DimGrid, DimBlock>>>(dA, dB, dC, m, n, p);

    cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

__global__ void MatMulKernel(float *A, float *B, float *C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * p + col;
    float value = 0.0;

    for (int i = 0; i < n; i++) value += A[row * n + i] * B[i * p + col];

    C[idx] = value;
}
