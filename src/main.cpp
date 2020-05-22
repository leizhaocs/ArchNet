#include "includes.h"

// data and network
bool use_gpu;
Net *net;
int num_train_images = 0;
int num_test_images = 0;
int num_image_chls = 0;
int num_image_rows = 0;
int num_image_cols = 0;
int num_classes = 0;
float *train_data_ptr;
float *train_label_ptr;
float *test_data_ptr;
float *test_label_ptr;
float *weights_ptr;

/* implemented in cpp files in data folder */
void load_mnist();
void load_cifar10();
void load_svhn();
void load_debug();

/* implemented in Config */
Params *build_network(const char *config, int &num_layers, int &num_epochs, float &lr_begin, float &lr_decay,
                      int &show_acc, bool &flip);

/* store weights file */
void store_weights(const char *weights, vector<vector<int>> total_weights)
{
    int num_weights = 0;
    for (int i = 0; i < total_weights.size(); i++)
    {
        num_weights += total_weights[i][0];
    }

    ofstream wghts(weights, ofstream::out);

    for (int i = 0; i < num_weights; i++)
    {
        wghts << weights_ptr[i] << endl;
    }

    wghts.close();
}

/* load weights file */
void load_weights(const char *weights, vector<vector<int>> total_weights)
{
    int num_weights = 0;
    for (int i = 0; i < total_weights.size(); i++)
    {
        num_weights += total_weights[i][0];
    }

    ifstream wghts(weights, ifstream::in);

    weights_ptr = new float[num_weights];
    for (int i = 0; i < num_weights; i++)
    {
        wghts >> weights_ptr[i];
    }

    wghts.close();
}

/* randomly initializing weights */
void random_weights(vector<vector<int>> total_weights)
{
    int num_weights = 0;
    for (int i = 0; i < total_weights.size(); i++)
    {
        num_weights += total_weights[i][0];
    }

    default_random_engine generator (105);
    normal_distribution<double> distribution(0.0, 1.0);

    weights_ptr = new float[num_weights];
    for (int i = 0; i < num_weights; i++)
    {
        weights_ptr[i] = distribution(generator) * 0.1;
    }
}

/* main function */
int main(int argc, char *argv[])
{
    DataType a = 2.789;
    DataType b = -1.4;
    DataType c = a * b;
    printf("%f\n", float(c));
    if (argc < 3)
    {
        cout<<"Usage:"<<endl;
        cout<<"    Train mode: ./nn train <dataset name> <network cfg file> <load weights file[null]> <save weights file[null]>"<<endl;
        cout<<"    Test  mode: ./nn test  <dataset name> <network cfg file> <load weights file>"<<endl;
        exit(0);
    }

    use_gpu = !find_arg(argc, argv, "-cpu");

    cout<<"Loading data..."<<endl;

    if (strcmp(argv[2], "mnist") == 0)
    {
        load_mnist();
    }
    else if (strcmp(argv[2], "cifar10") == 0)
    {
        load_cifar10();
    }
    else if (strcmp(argv[2], "svhn") == 0)
    {
        load_svhn();
    }
    else if (strcmp(argv[2], "debug") == 0)
    {
        load_debug();
    }
    else
    {
        Assert(false, "Unsupported dataset.");
    }

    cout<<"Building network..."<<endl;

    int num_layers;
    int num_epochs;
    float lr_begin;
    float lr_decay;
    int show_acc;
    bool flip;

    Params *layers = build_network(argv[3], num_layers, num_epochs, lr_begin, lr_decay, show_acc, flip);

    net = new Net(num_epochs, lr_begin, lr_decay, show_acc, flip);
    net->initLayers(layers, num_layers);

    cout<<"Setting data..."<<endl;

    net->readData(train_data_ptr, train_label_ptr, test_data_ptr, test_label_ptr,
                  num_train_images, num_test_images, num_image_chls, num_image_rows, num_image_cols, num_classes);

    if (strcmp(argv[1], "train") == 0)
    {
        if (strcmp(argv[4], "null") == 0)
        {
            cout<<"Initializing weights..."<<endl;

            vector<vector<int>> total_weights = net->getNumWeights();
            random_weights(total_weights);
            net->initWeights(weights_ptr);
        }
        else
        {
            cout<<"Loading weights..."<<endl;

            vector<vector<int>> total_weights = net->getNumWeights();
            load_weights(argv[4], total_weights);
            net->initWeights(weights_ptr);
        }

        MemoryMonitor::instance()->printCpuMemory();
#if GPU == 1
        MemoryMonitor::instance()->printGpuMemory();
#endif

        cout<<"Training..."<<endl;

        net->train();

        if (strcmp(argv[5], "null") != 0)
        {
            cout<<"Saving weights..."<<endl;

            vector<vector<int>> total_weights = net->getNumWeights();
            net->getWeights(weights_ptr);
            store_weights(argv[5], total_weights);
        }
    }
    else if (strcmp(argv[1], "test") == 0)
    {
        cout<<"Loading weights..."<<endl;

        vector<vector<int>> total_weights = net->getNumWeights();
        load_weights(argv[4], total_weights);
        net->initWeights(weights_ptr);

        MemoryMonitor::instance()->printCpuMemory();
#if GPU == 1
        MemoryMonitor::instance()->printGpuMemory();
#endif

        cout<<"Classifying..."<<endl;

        net->classify();
    }
    else
    {
        Assert(false, "Neither train nor test");
    }

    delete [] layers;
    delete [] weights_ptr;
    delete [] train_data_ptr;
    delete [] train_label_ptr;
    delete [] test_data_ptr;
    delete [] test_label_ptr;
    delete net;

    return 0;
}
