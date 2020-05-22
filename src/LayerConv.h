#ifndef _LAYER_CONV_H__
#define _LAYER_CONV_H__

#include "includes.h"

/* convolution layer */
class LayerConv : public Layer
{
public:
    /* constructor */
    LayerConv(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerConv();

    /* forward propagation */
    void cpu_forward(int realBatchSize, bool train);

    /* backward propagation */
    void cpu_backward(int realBatchSize);

    /* update weights and biases */
    void cpu_update(int realBatchSize, float lr);

#if GPU == 1
    /* forward propagation */
    void gpu_forward(int realBatchSize, bool train);

    /* backward propagation */
    void gpu_backward(int realBatchSize);

    /* update weights and biases */
    void gpu_update(int realBatchSize, float lr);
#endif

    /* initialize weights */
    void initWeights(float *weights, int &offset);

    /* get weights */
    void getWeights(float *weights, int &offset);

    /* get number of weights in this layer */
    std::vector<int> getNumWeights();

private:
    int num_filters_;                        // number of filters
    int filter_h_;                           // filter height
    int filter_w_;                           // filter width

    int stride_h_;                           // stride height
    int stride_w_;                           // stride width

    int padding_h_;                          // padding height
    int padding_w_;                          // padding width

    Tensor<DataType> *filters_;              // filters (f,c,h,w)
    Tensor<DataType> *biases_;               // biases (f)

    Tensor<DataType> *grad_filters_;         // gradients of filters (f,c,h,w)
    Tensor<DataType> *grad_biases_;          // gradients of biases (f)
};

#endif
