#ifndef __CUDA_UTIL__
#define __CUDA_UTIL__

#include "includes.h"

/* dropout */
void dropout_gpu(DataType *input, DataType *output, int n, int batch, float *rand, float rate);

/* backward of dropout */
void backward_dropout_gpu(DataType *backward_input, DataType *backward_output, float *rand, float rate, int n, int batch);

/* max pooling */
void maxpool_gpu(DataType *input, DataType *output, int *index, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w, int batch);

/* backward of maxpool */
void backward_maxpool_gpu(DataType *backward_input, DataType *backward_output, int *index, int in_h, int in_w, int in_c, int out_h, int out_w,
    int stride_h, int stride_w, int filter_h, int filter_w, int padding_h, int padding_w, int batch);

/* relu */
void relu_gpu(DataType *input, DataType *output, int n);

/* backward of relu */
void backward_relu_gpu(DataType *backward_input, DataType *forward_output, DataType *backward_output, int n);

/* sigmoid */
void sigmoid_gpu(DataType *input, DataType *output, int n);

/* backward of sigmoid */
void backward_sigmoid_gpu(DataType *backward_input, DataType *forward_output, DataType *backward_output, int n);

/* softmax */
void softmax_gpu(DataType *input, DataType *output, int n, int batch);

/* backward of softmax */
void backward_softmax_gpu(DataType *backward_input, DataType *forward_output, DataType *backward_output, int n, int batch);

/* grid size */
dim3 cuda_gridsize(int n);

/* check cuda error */
void check_cuda_error();

/* check cublas error */
void check_cublas_error(cublasStatus_t status);

/* get blas handle */
cublasHandle_t blas_handle();

#endif
