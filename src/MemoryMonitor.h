#ifndef __MEMORY_MONITOR_H__
#define __MEMORY_MONITOR_H__

#include "includes.h"

/* track allocated memory on cpu and gpu */
class MemoryMonitor
{
public:
    /* generate static instance */
    static MemoryMonitor *instance()
    {
        static MemoryMonitor *monitor = new MemoryMonitor();
        return monitor;
    }

    /* constructor */
    MemoryMonitor();

    /* malloc cpu memory */
    void cpuMalloc(void **hostPtr, int size);

    /* free cpu memory */
    void freeCpuMemory(void *ptr);

    /* print total malloc cpu memory */
    void printCpuMemory();

#if GPU == 1
    /* malloc gpu memory */
    void gpuMalloc(void **devPtr, int size);

    /* free gpu memory */
    void freeGpuMemory(void *ptr);

    /* print total malloc gpu memory */
    void printGpuMemory();
#endif

private:
    int cpuMemory;               // total malloc cpu memory
    map<void*, int> cpuPoint;    // track each allocated trunk of cpu memory
#if GPU == 1
    int gpuMemory;               // total malloc gpu memory
    map<void*, int> gpuPoint;    // track each allocated trunk of gpu memory
#endif
};

#endif
