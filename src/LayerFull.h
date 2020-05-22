#ifndef _LAYER_FULL_H__
#define _LAYER_FULL_H__

#include "includes.h"

/* fully connected layer */
class LayerFull : public Layer
{
public:
    /* constructor */
    LayerFull(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerFull();

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
    Tensor<DataType> *weights_;        // weights (prev_layer->sample_size_, sample_size_)
    Tensor<DataType> *biases_;         // biases (sample_size_)

    Tensor<DataType> *grad_weights_;   // gradients of weights (prev_layer->sample_size_, sample_size_)
    Tensor<DataType> *grad_biases_;    // gradients of biases (sample_size_)
};

#endif
