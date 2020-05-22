#include "includes.h"

__global__ void add_bias_kernel(DataType *output, DataType *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch)
    {
        return;
    }

    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

/* add biases in gpu */
void add_bias_gpu(DataType *output, DataType *biases, int batch, int n, int size)
{
    int num = n*size*batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_cuda_error();
}

__global__ void backward_bias_conn_kernel(DataType *grad_biases, DataType *delta, int batch, int n)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n)
    {
        return;
    }

    DataType sum = 0;
    for (int b = 0; b < batch; b++)
    {
        int i = b*n + index;
        sum += delta[i];
    }
    grad_biases[index] += sum;
}

__global__ void backward_bias_kernel(DataType *grad_biases, DataType *delta, int batch, int n, int size)
{
    __shared__ DataType part[BLOCK];

    int filter = blockIdx.x;
    int p = threadIdx.x;
    DataType sum = 0;
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < size; i += BLOCK)
        {
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0)
    {
        for(int i = 0; i < BLOCK; ++i)
        {
            grad_biases[filter] += part[i];
        }
    }
}

/* calculate gradient of biases in gpu */
void backward_bias_gpu(DataType *grad_biases, DataType *delta, int batch, int n, int size)
{
    if(size == 1)
    {
        backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>(grad_biases, delta, batch, n);
    }
    else
    {
        backward_bias_kernel<<<n, BLOCK>>>(grad_biases, delta, batch, n, size);
    }
    check_cuda_error();
}

__global__ void axpy_kernel(int N, DataType ALPHA, DataType *X,  DataType *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        Y[i] += ALPHA*X[i];
    }
}

/* Y += ALPHA * X */
void axpy_gpu(int N, DataType ALPHA, DataType *X, DataType *Y)
{
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, Y);
    check_cuda_error();
}

__global__ void clear_kernel(int N, DataType *X)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        X[i] = 0;
    }
}

/* clear all elements to 0 */
void clear_gpu(int N, DataType *X)
{
    clear_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X);
    check_cuda_error();
}

__global__ void random_kernel(int N, float *X)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        curandState state;
        curand_init(clock64(), i, 0, &state);
        X[i] = curand_uniform(&state);
    }
}

/* set all elements to random number between [0, 1] */
void random_gpu(int N, DataType *X)
{
    random_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X);
    check_cuda_error();
}
