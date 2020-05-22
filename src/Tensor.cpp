#include "includes.h"

/* constructor */
template<typename T>
Tensor<T>::Tensor(int n, int c, int h, int w)
{
    n_ = n;
    c_ = c;
    h_ = h;
    w_ = w;
    size_ = n * c * h * w;

    MemoryMonitor::instance()->cpuMalloc((void**)&data_cpu_, size_*sizeof(T));
#if GPU == 1
    if (use_gpu)
    {
        MemoryMonitor::instance()->gpuMalloc((void**)&data_gpu_, size_*sizeof(T));
    }
#endif
}

/* destructor */
template<typename T>
Tensor<T>::~Tensor()
{
    MemoryMonitor::instance()->freeCpuMemory(data_cpu_);
#if GPU == 1
    if (use_gpu)
    {
        MemoryMonitor::instance()->freeGpuMemory(data_gpu_);
    }
#endif
}

/* get data element */
template<typename T>
T &Tensor<T>::data(int i)
{
    return data_cpu_[i];
}
template<typename T>
T &Tensor<T>::data(int n, int i)
{
    int index = n*c_*h_*w_ + i;
    return data_cpu_[index];
}
template<typename T>
T &Tensor<T>::data(int n, int c, int h, int w)
{
    int index = ((n*c_ + c)*h_ + h)*w_ + w;
    return data_cpu_[index];
}

#if GPU == 1
/* move data from cpu to gpu */
template<typename T>
void Tensor<T>::toGpu()
{
    cudaError_t cudaStat = cudaMemcpy(data_gpu_, data_cpu_, size_*sizeof(T), cudaMemcpyHostToDevice);
    Assert(cudaStat == cudaSuccess, "To gpu data upload failed.");
}

/* move data from gpu to cpu */
template<typename T>
void Tensor<T>::toCpu()
{
    cudaError_t cudaStat = cudaMemcpy(data_cpu_, data_gpu_, size_*sizeof(T), cudaMemcpyDeviceToHost);
    Assert(cudaStat == cudaSuccess, "To cpu data download failed.");
}
#endif

/* get total number of elements */
template<typename T>
int Tensor<T>::size()
{
    return size_;
}

/* get cpu data pointer */
template<typename T>
T *Tensor<T>::getCpuPtr()
{
    return data_cpu_;
}

#if GPU == 1
/* get cpu data pointer */
template<typename T>
T *Tensor<T>::getGpuPtr()
{
    return data_gpu_;
}
#endif

template class Tensor<float>;
template class Tensor<int>;
#if STOCHASTIC == 1 || FIXEDPOINT == 1
template class Tensor<DataType>;
#endif
