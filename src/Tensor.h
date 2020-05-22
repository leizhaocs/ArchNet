#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "includes.h"

/* tensor */
template<typename T>
class Tensor
{
public:
    /* constructor */
    Tensor(int n, int c, int h, int w);

    /* destructor */
    ~Tensor();

    /* get data element */
    T &data(int i);
    T &data(int n, int i);
    T &data(int n, int c, int h, int w);

#if GPU == 1
    /* move data from cpu to gpu */
    void toGpu();

    /* move data from gpu to cpu */
    void toCpu();
#endif

    /* get total number of elements */
    int size();

    /* get cpu data pointer */
    T *getCpuPtr();

#if GPU == 1
    /* get cpu data pointer */
    T *getGpuPtr();
#endif

private:
    T *data_cpu_;         // raw data on cpu
#if GPU == 1
    T *data_gpu_;         // raw data on gpu
#endif
    int n_;               // batch size
    int c_;               // channel
    int h_;               // height
    int w_;               // width
    int size_;            // total number of elements
};

#endif
