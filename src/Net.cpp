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

/* constructor */
Net::Net(int num_epochs, float lr, float lr_decay, int show_acc, bool flip)
    : epochs_(num_epochs)
    , lr_(lr)
    , lr_decay_(lr_decay)
    , show_acc_(show_acc)
    , flip_(flip)
{
}

/* destructor */
Net::~Net()
{
    int layers_num = layers_.size();
    for (int i = 0; i < layers_num; ++i)
    {
        delete layers_[i];
    }
    layers_.clear();
    delete train_pred_;
    delete test_pred_;
}

/* read test data */
void Net::readData(float *train_data, float *train_labels, float *test_data, float *test_labels,
                  int n_train, int n_test, int c, int h, int w, int classes)
{
    train_data_ = train_data;
    train_labels_ = train_labels;
    test_data_ = test_data;
    test_labels_ = test_labels;
    n_train_ = n_train;
    n_test_ = n_test;
    c_ = c;
    h_ = h;
    w_ = w;
    classes_ = classes;
    train_pred_ = new float[n_train_*classes_];
    test_pred_ = new float[n_test_*classes_];
}

/* initialize all the layers in the network */
void Net::initLayers(Params *layer_params, int layers_num)
{
    Params *layer_param = &(layer_params[0]);
    batchSize_ = layer_params[0].getVectori("shape")[0];
    string layer_type = layer_param->getString("type");
    layers_.resize(layers_num);
    layers_[0] = new LayerInput(layer_param, NULL);
    for (int i = 1; i < layers_num; i++)
    {
        Layer *prev_layer = layers_[i-1];
        layer_param = &(layer_params[i]);
        layer_type = layer_param->getString("type");
        if (layer_type == "convolution")
        {
            layers_[i] = new LayerConv(layer_param, prev_layer);
        }
        else if (layer_type == "pool")
        {
            layers_[i] = new LayerPool(layer_param, prev_layer);
        }
        else if (layer_type == "full")
        {
            layers_[i] = new LayerFull(layer_param, prev_layer);
        }
        else if (layer_type == "activation")
        {
            layers_[i] = new LayerAct(layer_param, prev_layer);
        }
        else if (layer_type == "batchnormalization")
        {
            layers_[i] = new LayerBN(layer_param, prev_layer);
        }
        else if (layer_type == "dropout")
        {
            layers_[i] = new LayerDrop(layer_param, prev_layer);
        }
        else
        {
            Assert(false, layer_type + " - unknown type of the layer");
        }
        printf("%s\n", layer_type.c_str());
        printf("%d  %d  %d  %d\n", layers_[i]->n_, layers_[i]->c_, layers_[i]->h_, layers_[i]->w_);
    }
}

/* initialize weights for all layers */
void Net::initWeights(float *weights)
{
    int offset = 0;
    for (int i = 0; i < layers_.size(); i++)
    {
        layers_[i]->initWeights(weights, offset);
    }
}

/* get weights from all layers */
void Net::getWeights(float *weights)
{
    int offset = 0;
    for (int i = 0; i < layers_.size(); i++)
    {
        layers_[i]->getWeights(weights, offset);
    }
}

/* get number of weights per layer, only return layers with weights, 0: real 1; binary */
vector<vector<int>> Net::getNumWeights()
{
    vector<vector<int>> total_weights;
    for (int i = 0; i < layers_.size(); i++)
    {
        vector<int> num_weights = layers_[i]->getNumWeights();

        if (layers_[i]->type_ == "convolution")
        {
            vector<int> temp;
            for (int j = 0; j < num_weights.size(); j++)
            {
                temp.push_back(num_weights[j]);
            }
            total_weights.push_back(temp);
        }
        else if (layers_[i]->type_ == "full")
        {
            vector<int> temp;
            for (int j = 0; j < num_weights.size(); j++)
            {
                temp.push_back(num_weights[j]);
            }
            total_weights.push_back(temp);
        }
        else if (layers_[i]->type_ == "batchnormalization")
        {
            vector<int> temp;
            for (int j = 0; j < num_weights.size(); j++)
            {
                temp.push_back(num_weights[j]);
            }
            total_weights.push_back(temp);
        }
    }
    return total_weights;
}

/* train */
void Net::train()
{
    int numbatches = DIVUP(n_train_, batchSize_);

    index_.clear();
    for (int i = 0; i < n_train_; i++)
    {
        index_.push_back(i);
    }

    auto start = chrono::high_resolution_clock::now();

    for (int e = 0; e < epochs_; e++)
    {
        printf("Epoch: %d/%d   learning rate:%f\n", e+1, epochs_, lr_);

        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        shuffle(index_.begin(), index_.end(), default_random_engine(seed));
        int offset = 0;
        loss_ = 0;

        auto epoch_start = chrono::high_resolution_clock::now();

        for (int i = 0; i < numbatches; i++)
        {
            int realBatchSize = min(n_train_ - offset, batchSize_);

            initForward(train_data_, offset, realBatchSize);
            forward(realBatchSize, true);

            initBackward(train_labels_, offset, realBatchSize);
            backward(realBatchSize);
            update(realBatchSize);

            offset += realBatchSize;
        }

        auto epoch_end = chrono::high_resolution_clock::now();
        chrono::duration<double> epoch_diff = epoch_end-epoch_start;

        lr_ -= lr_decay_;
        loss_ /= n_train_;

        if (show_acc_ == 0)
        {
            printf("Loss: %f    time: %fs\n", loss_, epoch_diff.count());
        }
        else if (show_acc_ == 1)
        {
            float test_acc = validate(test_data_, test_labels_, n_test_, test_pred_);
            printf("Loss: %f    time: %fs    test acc: %f\n", loss_, epoch_diff.count(), test_acc);
        }
        else if (show_acc_ == 2)
        {
            float train_acc = validate(train_data_, train_labels_, n_train_, train_pred_);
            float test_acc = validate(test_data_, test_labels_, n_test_, test_pred_);
            printf("Loss: %f    time: %fs    test acc: %f    train acc: %f\n", loss_, epoch_diff.count(), test_acc, train_acc);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end-start;

    printf("================================\n");
    printf("        Training Results        \n");
    printf("time: %f s\n", diff.count());
}

/* inference */
void Net::classify()
{
    auto start = chrono::high_resolution_clock::now();

    float accuracy = validate(test_data_, test_labels_, n_test_, test_pred_);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end-start;

    printf("================================\n");
    printf("           Test Results         \n");
    printf("acc: %f    time: %f s\n", accuracy, diff.count());
}

/* called before every forward propagation */
void Net::initForward(float *input, int offset, int realBatchSize)
{
    for (int i = offset, n = 0; i < offset+realBatchSize; i++, n++)
    {
        for (int c = 0; c < c_; c++)
        {
            for (int h = 0; h < h_; h++)
            {
                for (int w = 0; w < w_; w++)
                {
                    int index = ((index_[i]*c_ + c)*h_ + h)*w_ + w;;
                    layers_[0]->forwardTensor_->data(n, c, h, w) = input[index];
                }
            }
        }
    }
#if GPU == 1
    if (use_gpu)
    {
        layers_[0]->forwardTensor_->toGpu();
    }
#endif
}

/* called before every backward propagation */
void Net::initBackward(float *groudtruth, int offset, int realBatchSize)
{
#if GPU == 1
    if (use_gpu)
    {
        layers_.back()->forwardTensor_->toCpu();
    }
#endif

    int k = 0;
    for (int i = offset; i < offset+realBatchSize; i++)
    {
        for (int j = 0; j < classes_; j++)
        {
            float l = log((float)layers_.back()->forwardTensor_->data(k, j)+EPSILON) * groudtruth[index_[i]*classes_ + j] * -1;
            layers_.back()->backwardTensor_->data(k, j) = l;
            loss_ += l;
        }
        k++;
    }

#if GPU == 1
    if (use_gpu)
    {
        layers_.back()->backwardTensor_->toGpu();
    }
#endif
}

/* forward propagate through all layers */
void Net::forward(int realBatchSize, bool train)
{
    for (int i = 0; i < layers_.size(); i++)
    {
#if GPU == 1
        if (use_gpu)
        {
            layers_[i]->gpu_forward(realBatchSize, train);
        }
        else
        {
            layers_[i]->cpu_forward(realBatchSize, train);
        }
#else
        layers_[i]->cpu_forward(realBatchSize, train);
#endif
    }
}

/* backward propagate through all layers */
void Net::backward(int realBatchSize)
{
    for (int i = layers_.size()-1; i >= 0; i--)
    {
#if GPU == 1
        if (use_gpu)
        {
            layers_[i]->gpu_backward(realBatchSize);
        }
        else
        {
            layers_[i]->cpu_backward(realBatchSize);
        }
#else
        layers_[i]->cpu_backward(realBatchSize);
#endif
    }
}

/* update weights and biases */
void Net::update(int realBatchSize)
{
    for (int i = layers_.size()-1; i >= 0; i--)
    {
#if GPU == 1
        if (use_gpu)
        {
            layers_[i]->gpu_update(realBatchSize, lr_);
        }
        else
        {
            layers_[i]->cpu_update(realBatchSize, lr_);
        }
#else
        layers_[i]->cpu_update(realBatchSize, lr_);
#endif
    }
}

/* get the prediction of a batch */
void Net::getPrediction(float *pred, int realBatchSize)
{
#if GPU == 1
    if (use_gpu)
    {
        layers_.back()->forwardTensor_->toCpu();
    }
#endif
    datatype_to_float(realBatchSize*classes_, pred, layers_.back()->forwardTensor_->getCpuPtr());
}

/* validate */
float Net::validate(float *data, float *labels, int num_images, float *pred)
{
    int offset = 0;
    int numbatches = DIVUP(num_images, batchSize_);

    index_.clear();
    for (int i = 0; i < num_images; i++)
    {
        index_.push_back(i);
    }

    for (int i = 0; i < numbatches; i++)
    {
        int realBatchSize = min(num_images - offset, batchSize_);

        initForward(data, offset, realBatchSize);
        forward(realBatchSize, false);

        float *p = pred + offset*classes_;
        getPrediction(p, realBatchSize);

        offset += realBatchSize;
    }

    int correct = 0, wrong = 0;
    for (int i = 0; i < num_images; i++)
    {
        float pred_max = 0, max = 0;
        int pred_digit = -1, digit = -2;
        for (int j = 0; j < 10; j++)
        {
            if (pred[i*10+j] > pred_max)
            {
                pred_max = pred[i*10+j];
                pred_digit = j;
            }
            if (labels[i*10+j] > max)
            {
                max = labels[i*10+j];
                digit = j;
            }
        }
        if (digit == pred_digit)
        {
            correct++;
        }
        else
        {
            wrong++;
        }
    }

    return ((float)correct)/num_images;
}
