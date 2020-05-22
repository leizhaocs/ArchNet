#include "includes.h"

/* constructor */
LayerInput::LayerInput(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "input";

    Assert(params->hasField("shape"), "Input Layer must have shape specified.");
    std::vector<int> shape = params->getVectori("shape");
    n_ = shape[0];
    c_ = shape[1];
    h_ = shape[2];
    w_ = shape[3];
    sample_size_ = c_ * h_ * w_;

    forwardTensor_ = new Tensor<DataType>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<DataType>(n_, c_, h_, w_);
}

/* destructor */
LayerInput::~LayerInput()
{
    delete forwardTensor_;
    delete backwardTensor_;
}

/* forward propagation */
void LayerInput::cpu_forward(int realBatchSize, bool train)
{
}

/* backward propagation */
void LayerInput::cpu_backward(int realBatchSize)
{
}

/* update weights and biases */
void LayerInput::cpu_update(int realBatchSize, float lr)
{
}

#if GPU == 1
/* forward propagation */
void LayerInput::gpu_forward(int realBatchSize, bool train)
{
}

/* backward propagation */
void LayerInput::gpu_backward(int realBatchSize)
{
}

/* update weights and biases */
void LayerInput::gpu_update(int realBatchSize, float lr)
{
}
#endif

/* initialize weights */
void LayerInput::initWeights(float *weights, int &offset)
{
}

/* get weights */
void LayerInput::getWeights(float *weights, int &offset)
{
}

/* get number of weights in this layer */
std::vector<int> LayerInput::getNumWeights()
{
    std::vector<int> num_weights{0};
    return num_weights;
}
