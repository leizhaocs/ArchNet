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

#ifndef _NET_H__
#define _NET_H__

#include "includes.h"

/* neural network */
class Net
{
public:
    /* constructor */
    Net(int num_epochs, float lr, float lr_decay, int show_acc, bool flip);

    /* destructor */
    ~Net();

    /* read data */
    void readData(float *train_data, float *train_labels, float *test_data, float *test_labels,
                  int n_train, int n_test, int c, int h, int w, int classes);

    /* initialize all the layers in the network */
    void initLayers(Params *layer_params, int layers_num);

    /* initialize weights for all layers */
    void initWeights(float *weights);

    /* get weights from all layers */
    void getWeights(float *weights);

    /* get number of weights per layer, only return layers with weights, 0: real 1: binary */
    vector<vector<int>> getNumWeights();

    /* train */
    void train();

    /* inference */
    void classify();

private:
    /* called before every forward propagation */
    void initForward(float *input, int offset, int realBatchSize);

    /* called before every backward propagation */
    void initBackward(float *groudtruth, int offset, int realBatchSize);

    /* forward propagate through all layers */
    void forward(int realBatchSize, bool train);

    /* backward propagate through all layers */
    void backward(int realBatchSize);

    /* update weights and biases */
    void update(int realBatchSize);

    /* get the prediction of a batch */
    void getPrediction(float *pred, int realBatchSize);

    /* validate */
    float validate(float *data, float *labels, int num_images, float *pred);

    int epochs_;                   // number of epochs
    float lr_;                     // learning rate
    float lr_decay_;               // decay of learning rate
    vector<Layer *> layers_;       // all the layers
    float *train_data_;            // training data (n,h,w,c)
    float *train_labels_;          // training labels one hot vectors
    float *train_pred_;            // predictions on train data
    float *test_data_;             // test data (n,h,w,c)
    float *test_labels_;           // test labels one hot vectors
    float *test_pred_;             // predictions on test data
    int n_train_;                  // number of samples in the training dataset
    int n_test_;                   // number of samples in the test dataset
    int c_;                        // channels of input
    int h_;                        // height of input
    int w_;                        // width of input
    int classes_;                  // number of classes
    int batchSize_;                // batch size
    vector<int> index_;            // used to shuffle data and labels in each epoch
    float loss_;                   // average loss of an epoch in training
    int show_acc_;                 // show accuracy in each epoch
    bool flip_;                    // flip image
};

#endif
