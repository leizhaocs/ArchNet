#include "includes.h"

/* constructor */
MemoryMonitor::MemoryMonitor()
{
    cpuMemory = 0;
#if GPU == 1
    gpuMemory = 0;
#endif
}

/* malloc cpu memory */
void MemoryMonitor::cpuMalloc(void **hostPtr, int size)
{
    cpuMemory += size;
    *hostPtr = (void *)malloc(size);
    memset(*hostPtr, 0, size);
    cpuPoint[*hostPtr] = size;
}

/* free cpu memory */
void MemoryMonitor::freeCpuMemory(void *ptr)
{
    if (cpuPoint.find(ptr) != cpuPoint.end())
    {
        cpuMemory -= cpuPoint[ptr];
        free(ptr);
        cpuPoint.erase(ptr);
    }
}

/* print total malloc cpu memory */
void MemoryMonitor::printCpuMemory()
{
    printf("total malloc cpu memory %fMb\n", cpuMemory / 1024.0f / 1024.0f);
}

#if GPU == 1
/* malloc gpu memory */
void MemoryMonitor::gpuMalloc(void **devPtr, int size)
{
    gpuMemory += size;
    cudaError_t error = cudaMalloc(devPtr, size);
    Assert(error == cudaSuccess, "Device memory allocation failed.");
    gpuPoint[*devPtr] = size;
}

/* free gpu memory */
void MemoryMonitor::freeGpuMemory(void *ptr)
{
    if (gpuPoint.find(ptr) != gpuPoint.end())
    {
        gpuMemory -= gpuPoint[ptr];
        cudaFree(ptr);
        gpuPoint.erase(ptr);
    }
}

/* print total malloc gpu memory */
void MemoryMonitor::printGpuMemory()
{
    printf("total malloc gpu memory %fMb\n", gpuMemory / 1024.0f / 1024.0f);
}
#endif
