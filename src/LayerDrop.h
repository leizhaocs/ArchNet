#ifndef _LAYER_DROP_H__
#define _LAYER_DROP_H__

#include "includes.h"

/* activation layer */
class LayerDrop : public Layer
{
public:
    /* constructor */
    LayerDrop(Params *params, Layer *prev_layer);

    /* destructor */
    ~LayerDrop();

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
    float rate_;                     // drop out rate
    Tensor<float> *dropoutTensor_;   // same size with forwardTensor (randomly generated)
};

#endif
