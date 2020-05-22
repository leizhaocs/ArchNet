#ifndef _LAYER_BN_H__
#define _LAYER_BN_H__

#include "includes.h"

/* batch normalization layer */
class LayerBN : public Layer
{
public:
    /* constructor */
    LayerBN(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerBN();

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
    int channels_;                   // the second dimension of input

    Tensor<DataType> *mean_;         // mean
    Tensor<DataType> *std_;          // std
    Tensor<DataType> *beta_;         // beta
    Tensor<DataType> *gamma_;        // gamma

    Tensor<DataType> *running_mean_; // running mean
    Tensor<DataType> *running_std_;  // running std
    Tensor<DataType> *grad_beta_;    // gradients of beta
    Tensor<DataType> *grad_gamma_;   // gradients of beta
};

#endif
