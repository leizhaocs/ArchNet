#include "includes.h"

/* constructor */
LayerDrop::LayerDrop(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "dropout";

    n_ = prev_layer->n_;
    c_ = prev_layer->c_;
    h_ = prev_layer->h_;
    w_ = prev_layer->w_;
    sample_size_ = prev_layer->sample_size_;

    forwardTensor_ = new Tensor<DataType>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<DataType>(n_, c_, h_, w_);

    Assert(params->hasField("rate"), "Dropout Layer must have dropout rate specified.");
    rate_ = params->getScalarf("rate");

    dropoutTensor_ = new Tensor<float>(n_, c_, h_, w_);
}

/* destructor */
LayerDrop::~LayerDrop()
{
    delete forwardTensor_;
    delete backwardTensor_;
    delete dropoutTensor_;
}

/* forward propagation */
void LayerDrop::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_->size(), forwardTensor_->getCpuPtr());

    if (train)
    {
        random(dropoutTensor_->size(), dropoutTensor_->getCpuPtr());
        for (int i = 0; i < forwardTensor_->size(); i++)
        {
            if (dropoutTensor_->data(i) < rate_)
            {
                forwardTensor_->data(i) = 0;
            }
            else
            {
                forwardTensor_->data(i) = prev_layer_->forwardTensor_->data(i);
            }
        }
    }
    else
    {
        for (int i = 0; i < forwardTensor_->size(); i++)
        {
            forwardTensor_->data(i) = prev_layer_->forwardTensor_->data(i);
        }
    }
}

/* backward propagation */
void LayerDrop::cpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getCpuPtr());

    for (int i = 0; i < prev_layer_->backwardTensor_->size(); i++)
    {
        if (dropoutTensor_->data(i) < rate_)
        {
            prev_layer_->backwardTensor_->data(i) = 0;
        }
        else
        {
            prev_layer_->backwardTensor_->data(i) = backwardTensor_->data(i);
        }
    }
}

/* update weights and biases */
void LayerDrop::cpu_update(int realBatchSize, float lr)
{
}

#if GPU == 1
/* forward propagation */
void LayerDrop::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_->size(), forwardTensor_->getGpuPtr());

    if (train)
    {
        random_gpu(dropoutTensor_->size(), dropoutTensor_->getGpuPtr());
        dropout_gpu(prev_layer_->forwardTensor_->getGpuPtr(), forwardTensor_->getGpuPtr(), sample_size_, realBatchSize,
            dropoutTensor_->getGpuPtr(), rate_);
    }
    else
    {
        cudaMemcpy(forwardTensor_->getGpuPtr(), prev_layer_->forwardTensor_->getGpuPtr(),
            forwardTensor_->size()*sizeof(DataType), cudaMemcpyDeviceToDevice);
    }
}

/* backward propagation */
void LayerDrop::gpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear_gpu(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getGpuPtr());

    backward_dropout_gpu(backwardTensor_->getGpuPtr(), prev_layer_->backwardTensor_->getGpuPtr(), dropoutTensor_->getGpuPtr(),
        rate_, sample_size_, realBatchSize);
}

/* update weights and biases */
void LayerDrop::gpu_update(int realBatchSize, float lr)
{
}
#endif

/* initialize weights */
void LayerDrop::initWeights(float *weights, int &offset)
{
}

/* get weights */
void LayerDrop::getWeights(float *weights, int &offset)
{
}

/* get number of weights in this layer */
std::vector<int> LayerDrop::getNumWeights()
{
    std::vector<int> num_weights{0};
    return num_weights;
}
