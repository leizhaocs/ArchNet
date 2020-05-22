#include "includes.h"

/* constructor */
LayerConv::LayerConv(Params *params, Layer *prev_layer)
{
    prev_layer_ = prev_layer;

    type_ = "convolution";

    Assert(params->hasField("filterSize"), "Convolutional Layer must have filter size specified.");
    std::vector<int> filterSize = params->getVectori("filterSize");
    num_filters_= filterSize[0];
    filter_h_ = filterSize[1];
    filter_w_ = filterSize[2];

    Assert(params->hasField("stride"), "Convolutional Layer must have stride specified.");
    std::vector<int> stride = params->getVectori("stride");
    stride_h_ = stride[0];
    stride_w_ = stride[1];

    Assert(params->hasField("padding"), "Convolutional Layer must have padding specified.");
    std::vector<int> padding = params->getVectori("padding");
    padding_h_ = padding[0];
    padding_w_ = padding[1];

    n_ = prev_layer_->n_;
    c_ = num_filters_;
    h_ = (prev_layer_->h_-filter_h_+2*padding_h_)/stride_h_ + 1;
    w_ = (prev_layer_->w_-filter_w_+2*padding_w_)/stride_w_ + 1;
    sample_size_ = c_ * h_ * w_;

    forwardTensor_ = new Tensor<DataType>(n_, c_, h_, w_);
    backwardTensor_ = new Tensor<DataType>(n_, c_, h_, w_);

    filters_ = new Tensor<DataType>(num_filters_, prev_layer_->c_, filter_h_, filter_w_);
    biases_ = new Tensor<DataType>(1, 1, 1, num_filters_);

    grad_filters_ = new Tensor<DataType>(num_filters_, prev_layer_->c_, filter_h_, filter_w_);
    grad_biases_ = new Tensor<DataType>(1, 1, 1, num_filters_);
}

/* destructor */
LayerConv::~LayerConv()
{
    delete forwardTensor_;
    delete backwardTensor_;
    delete filters_;
    delete biases_;
    delete grad_filters_;
    delete grad_biases_;
}

/* forward propagation */
void LayerConv::cpu_forward(int realBatchSize, bool train)
{
    clear(forwardTensor_->size(), forwardTensor_->getCpuPtr());

    int m = num_filters_;
    int n = h_ * w_;
    int k = filter_h_ * filter_w_ * prev_layer_->c_;

    Tensor<DataType> *atensor = new Tensor<DataType>(1, 1, k, n);
    for (int i = 0; i < realBatchSize; i++)
    {
        DataType *a = filters_->getCpuPtr();
        DataType *b = atensor->getCpuPtr();
        DataType *c = forwardTensor_->getCpuPtr() + i*sample_size_;

        DataType *im = prev_layer_->forwardTensor_->getCpuPtr() + i*prev_layer_->sample_size_;
        im2col_cpu(im, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, b);
        gemm(0, 0, m, n, k, a, k, b, n, c, n);
    }
    delete atensor;
    add_bias(forwardTensor_->getCpuPtr(), biases_->getCpuPtr(), realBatchSize, num_filters_, h_*w_);
}

/* backward propagation */
void LayerConv::cpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getCpuPtr());
    clear(grad_filters_->size(), grad_filters_->getCpuPtr());
    clear(grad_biases_->size(), grad_biases_->getCpuPtr());

    int m = num_filters_;
    int n = filter_h_ * filter_w_ * prev_layer_->c_;
    int k = h_ * w_;

    Tensor<DataType> *atensor = new Tensor<DataType>(n, 1, 1, k);
    for (int i = 0; i < realBatchSize; i++)
    {
        clear(atensor->size(), atensor->getCpuPtr());

        DataType *a = backwardTensor_->getCpuPtr() + i*sample_size_;
        DataType *b = atensor->getCpuPtr();
        DataType *c = grad_filters_->getCpuPtr();

        DataType *im = prev_layer_->forwardTensor_->getCpuPtr() + i*prev_layer_->sample_size_;
        im2col_cpu(im, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, b);
        gemm(0, 1, m, n, k, a, k, b, k, c, n);

        clear(atensor->size(), atensor->getCpuPtr());

        a = filters_->getCpuPtr();
        b = backwardTensor_->getCpuPtr() + i*sample_size_;
        c = atensor->getCpuPtr();

        gemm(1, 0, n, k, m, a, n, b, k, c, k);
        DataType *imd = prev_layer_->backwardTensor_->getCpuPtr() + i*prev_layer_->sample_size_;
        col2im_cpu(c, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, imd);
    }
    delete atensor;
    backward_bias(grad_biases_->getCpuPtr(), backwardTensor_->getCpuPtr(), realBatchSize, num_filters_, k);
}

/* update weights and biases */
void LayerConv::cpu_update(int realBatchSize, float lr)
{
    DataType step = -lr/realBatchSize;
    axpy(filters_->size(), step, grad_filters_->getCpuPtr(), filters_->getCpuPtr());
    axpy(biases_->size(), step, grad_biases_->getCpuPtr(), biases_->getCpuPtr());
}

#if GPU == 1
/* forward propagation */
void LayerConv::gpu_forward(int realBatchSize, bool train)
{
    clear_gpu(forwardTensor_->size(), forwardTensor_->getGpuPtr());

    int m = num_filters_;
    int n = h_ * w_;
    int k = filter_h_ * filter_w_ * prev_layer_->c_;

    Tensor<DataType> *atensor = new Tensor<DataType>(n, 1, 1, k);
    for (int i = 0; i < realBatchSize; i++)
    {
        DataType *a = filters_->getGpuPtr();;
        DataType *b = atensor->getGpuPtr();
        DataType *c = forwardTensor_->getGpuPtr() + i*sample_size_;

        DataType *im = prev_layer_->forwardTensor_->getGpuPtr() + i*prev_layer_->sample_size_;
        im2col_gpu(im, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, b);
        gemm_gpu(0, 0, m, n, k, a, k, b, n, c, n);
    }
    delete atensor;
    add_bias_gpu(forwardTensor_->getGpuPtr(), biases_->getGpuPtr(), realBatchSize, num_filters_, h_*w_);
}

/* backward propagation */
void LayerConv::gpu_backward(int realBatchSize)
{
    if (prev_layer_->type_ == "input")
    {
        return;
    }

    clear_gpu(prev_layer_->backwardTensor_->size(), prev_layer_->backwardTensor_->getGpuPtr());
    clear_gpu(grad_filters_->size(), grad_filters_->getGpuPtr());
    clear_gpu(grad_biases_->size(), grad_biases_->getGpuPtr());

    int m = num_filters_;
    int n = filter_h_ * filter_w_ * prev_layer_->c_;
    int k = h_ * w_;

    Tensor<DataType> *atensor = new Tensor<DataType>(1, 1, k, n);
    for (int i = 0; i < realBatchSize; i++)
    {
        clear_gpu(atensor->size(), atensor->getGpuPtr());

        DataType *a = backwardTensor_->getGpuPtr() + i*sample_size_;
        DataType *b = atensor->getGpuPtr();
        DataType *c = grad_filters_->getGpuPtr();

        DataType *im = prev_layer_->forwardTensor_->getGpuPtr() + i*prev_layer_->sample_size_;
        im2col_gpu(im, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, b);
        gemm_gpu(0, 1, m, n, k, a, k, b, k, c, n);

        clear_gpu(atensor->size(), atensor->getGpuPtr());

        a = filters_->getGpuPtr();
        b = backwardTensor_->getGpuPtr() + i*sample_size_;
        c = atensor->getGpuPtr();

        gemm_gpu(1, 0, n, k, m, a, n, b, k, c, k);
        DataType *imd = prev_layer_->backwardTensor_->getGpuPtr() + i*prev_layer_->sample_size_;
        col2im_gpu(c, prev_layer_->c_, prev_layer_->h_, prev_layer_->w_, h_, w_,
                   filter_h_, filter_w_, stride_h_, stride_w_, padding_h_, padding_w_, imd);
    }
    delete atensor;
    backward_bias_gpu(grad_biases_->getGpuPtr(), backwardTensor_->getGpuPtr(), realBatchSize, num_filters_, k);
}

/* update weights and biases */
void LayerConv::gpu_update(int realBatchSize, float lr)
{
    DataType step = -lr/realBatchSize;
    axpy_gpu(filters_->size(), step, grad_filters_->getGpuPtr(), filters_->getGpuPtr());
    axpy_gpu(biases_->size(), step, grad_biases_->getGpuPtr(), biases_->getGpuPtr());
}
#endif

/* initialize weights */
void LayerConv::initWeights(float *weights, int &offset)
{
    float_to_datatype(filters_->size(), filters_->getCpuPtr(), weights+offset);
    offset += filters_->size();

    float_to_datatype(biases_->size(), biases_->getCpuPtr(), weights+offset);
    offset += biases_->size();

#if GPU == 1
    if (use_gpu)
    {
        filters_->toGpu();
        biases_->toGpu();
    }
#endif
}

/* get weights */
void LayerConv::getWeights(float *weights, int &offset)
{
#if GPU == 1
    if (use_gpu)
    {
        filters_->toCpu();
        biases_->toCpu();
    }
#endif

    datatype_to_float(filters_->size(), weights+offset, filters_->getCpuPtr());
    offset += filters_->size();

    datatype_to_float(biases_->size(), weights+offset, biases_->getCpuPtr());
    offset += biases_->size();
}

/* get number of weights in this layer */
std::vector<int> LayerConv::getNumWeights()
{
    int nf = filters_->size();
    int nb = biases_->size();
    std::vector<int> num_weights;
    num_weights.push_back(nf+nb);
    num_weights.push_back(nf);
    num_weights.push_back(nb);
    return num_weights;
}
