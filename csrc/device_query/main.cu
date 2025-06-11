// chapter 4 of pmpp
#include <stdio.h>

int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("This computer has: %d cuda capable devices\n", devCount);
    cudaDeviceProp devProp;
    for (unsigned int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        printf("\nname: %s\n", devProp.name);
        printf("Compute Capability: %d.%d\n", devProp.major, devProp.minor);
        printf("maxThreadsPerBlock: %d\n", devProp.maxThreadsPerBlock);
        printf("multiProcessorCount, aka number of SMs: %d\n", devProp.multiProcessorCount);
        printf("clockRate: %.2f GHz\n", devProp.clockRate * 1e-6f);
        printf("maxGridSize, aka maximum number of blocks: (%d, %d, %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("maxThreadsDim, aka maximum number of threads per block: (%d, %d, %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("regsPerBlock, aka number registers per SMs: %d\n", devProp.regsPerBlock);
        printf("warpSize: %d\n", devProp.warpSize);
    }
}