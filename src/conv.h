#ifndef _CONV_H_
#define _CONV_H_

#include "includes.h"

void im2col_cpu(DataType* data_im, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, DataType* data_col);

void col2im_cpu(DataType *data_col, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, DataType *data_im);

#if GPU == 1
void im2col_gpu(DataType *data_im, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, DataType *data_col);

void col2im_gpu(DataType *data_col, int input_c, int input_h, int input_w, int output_h, int output_w,
                int filter_h, int filter_w, int stride_h, int stride_w, int padding_h, int padding_w, DataType *data_im);
#endif

#endif
