/**********************************************************************
 *
 * Copyright Lei Zhao.
 * contact: leizhao0403@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **********************************************************************/

#include "includes.h"

/* add biases */
void add_bias(DataType *output, DataType *biases, int batch, int n, int size)
{
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < size; j++)
            {
                output[(b * n + i) * size + j] += biases[i];
            }
        }
    }
}

/* sum up all the elements in an array */
DataType sum_array(DataType *a, int n)
{
    DataType sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    return sum;
}

/* backward of add_bias */
void backward_bias(DataType *grad_biases, DataType *delta, int batch, int n, int size)
{
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < n; i++)
        {
            grad_biases[i] += sum_array(delta + size * (i + b * n), size);
        }
    }
}

/* Y += ALPHA * X */
void axpy(int N, DataType ALPHA, DataType *X, DataType *Y)
{
    for (int i = 0; i < N; i++)
    {
        Y[i] += ALPHA * X[i];
    }
}

/* clear all elements to 0 */
void clear(int N, DataType *X)
{
    for (int i = 0; i < N; i++)
    {
        X[i] = 0;
    }
}

/* set all elements to random number between [0, 1] */
void random(int N, float *X)
{
    for (int i = 0; i < N; i++)
    {
        X[i] = rand()/RAND_MAX;
    }
}

/* copy from float array to DataType array */
void float_to_datatype(int n, DataType *d, float *f)
{
    for (int i = 0; i < n; i++)
    {
        d[i] = f[i];
    }
}

/* copy from DataType array to float array */
void datatype_to_float(int n, float *f, DataType *d)
{
    for (int i = 0; i < n; i++)
    {
        f[i] = (float)d[i];
    }
}
