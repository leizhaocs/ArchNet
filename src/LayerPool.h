#ifndef _LAYER_POOL_H__
#define _LAYER_POOL_H__

#include "includes.h"

class LayerPool : public Layer
{
public:
    /* constructor */
    LayerPool(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerPool();

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
    std::string pool_type_;       // pooling type

    int filter_h_;                // filter height
    int filter_w_;                // filter width

    int stride_h_;                // stride height
    int stride_w_;                // stride width

    int padding_h_;               // padding height
    int padding_w_;               // padding width

    Tensor<int> *indexTensor_;    // recording the index of the maximum neuron in the max pooling window
};

#endif
