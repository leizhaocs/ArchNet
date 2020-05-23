# ArchNet
A neural network for architecture research

**Why another neural network framework?**
The primary target users of ArchNet are computer architecture researchers who need to modify lower level implementations of neural network computing. However, as speed and configurability are carefully considered since the begining of the development, ArchNet can also be used in general neural network researches.

In architecture research, researchers often need to make modifications at the lowest level of neural network computing, e.g., modify the CUDA kernels for convolutions, change all computations into fixed point format for quantization, use stochastic computing for cheap circuit implementation, etc. However, current existing neural networks do not provide an easy way to make these modifications feasible. Because their primary focus is speed and flexibility, it is very hard to dig too deep into the code. On the contrary, ArchNet only implements the most common layers in CNNs, all the codes are very clean and easy to understand (with sufficient comments), while not sacrificing the speed of execution.

 **Features of ArchNet**

- Support both training and inference. 

- Configurability. All layers are parameterized. The network structure can be easily configured by stacking layers in a config file.

- Written in C++. To achieve higher speed and more exposure to the underlying implementation, ArchNet is written in C++.

- CUDA acceleration. ArchNet provides an easy way to modify the CUDA kernels for all layers. Some kernels may not be fully optimized in order for simplicity and easy to understand.

- Fixed point computing. Using fixed point format instead of floating point is a very common method in architecture research for faster computing and quantization. ArchNet uses overloaded operators for all CNN computings, it is very easy to change all computing into fixed point format by simply set a flag during compilation.

- Stochastic computing. Stochastic computing is becoming popular because its hardware implementation can easily achieve lower power and smaller area footprint. However, stochastic computing also poses challenges on neural network's classification accuracy. ArchNet makes evaluating different stochastic computing schemes in the easiest way. Researchers only need to modify one header file according to their custom stochastic computing scheme and set the flag during compilation, all computations in the CNN will use the custom stochastic computing scheme instead of floating point computation.

- Homomorphic encryption. As CNNs are more and more widely used in commercial products, user's privacy concern is becoming a issue these days. ArchNet integrates CKKS homomorphic encryption into CNN computing.

 **Compilation**

In ArchNet folder, type make to compile.

There are three flags in Makefile for different computing schemes.

- **GPU** Set to 1 to use cuda during compilation

- **STOCHASTIC** Set to 1 to use stochastic computing.

- **FIXEDPOINT** Set to 1 to use fixed point format.

- **HE** Set to 1 to use homomorphic encryption.

 **Run**

For training:

    ./nn train <dataset name> <network cfg file> <load weights file[null]> <save weights file[null]> -cpu

        e.g. ./nn train mnist cfg/mnist_cnn.cfg null weights/mnist.weights

        You can also perform training based on some exsiting weights by changing null to some exiting weights file, e.g.
        ./nn train mnist cfg/mnist_cnn.cfg weights/mnist.weights_old weights/mnist.weights_new

        By setting -cpu flag, the execution will only use cpu even the code is compiled with GPU flag set in the Makefile.

For testing:

    ./nn test  <dataset name> <network cfg file> <load weights file>

        e.g. ./nn train mnist cfg/mnist_cnn.cfg weights/mnist.weights

        By setting -cpu flag, the execution will only use cpu even the code is compiled with GPU flag set in the Makefile.

 **Dataset**

To download the dataset, go to the data folder and execute the python file.

 **Layers**

Config file format:

    To write a config file, the format must strictly follow the instructions in this
    section. It is recommended to modify from an existing config file in cfg folder.

    [global]
    layers: total number of layers in the network.
    epochs: number of epochs in training.
    lr_begin: the learning rate in the first epoch. Learning rate decreases linearly during training.
    lr_end: the learning rate in the last epoch. Learning rate decreases linearly during training.
    show_acc: 0 - 2. 0: print the least information, 2: print the most information.
    flip: randomly flip the input image to augment dataset.

    [input]
    shape: four integers seperated by space, meaning batch size, channels, height, width.

    [convolution]
    filterSize: three integers seperated by space, meaning number of filters, height of each filter, width of each filter.
    stride: two integers seperated by space, meaning the step along height and width dimensions of the input.
    padding: two integers seperated by space, meaning the number of 0 paddings on each side of the height and width dimensions of the input. For example, if set to 1 and 2, there will be 1 row of 0s padding on the top and bottom of the image, and two rows of 0s padding on the left and right of the image.

    [full]
    length: one integer indicating the number of output neurons.

    [activation]
    nonlinear: a string one of "relu", "sigmoid" and "softmax".

    [pool]
    poolType: a string one of "max" and "mean".
    filterSize: two integers seperated by space, meaning pooling window size along height and width dimensions of the input.
    stride: two integers seperated by space, meaning the step along height and width dimensions of the input.
    padding: two integers seperated by space, meaning the number of 0 paddings on each side of the height and width dimensions of the input. For example, if set to 1 and 2, there will be 1 row of 0s padding on the top and bottom of the image, and two rows of 0s padding on the left and right of the image.

    [batchnormalization]
    Currently not implemented.

