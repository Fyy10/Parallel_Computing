#include "cuda.h"
#include "stdio.h"

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Num of devices: %d\n", deviceCount);
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has capability %d.%d\n", device, deviceProp.major, deviceProp.minor);
    }
    return 0;
}
