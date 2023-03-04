# neural-image-recog

I will implement a deep convolutional neural network (CNN) for image classification. We will use a customized dataset with 5 classes (i.e. face, airplane, dog, car, tree), and the dataset contains a thousand 30x30 color images per class. This dataset is selected from AffectNet, ImageNet, and CIFAR-10. This "toy" dataset is small enough to run on a CPU so that you can taste deep learning with limited resources.

## Neural Network
I have created a simple neural network with one hidden layer (NN class in the answer.py file). Train the model by using the command python main.py --model NN.

## Simple Convolution Neural Network
I have created a simple convolutional neural network with one hidden convolutional layer and one hidden fully-connected layer (SimpleCNN class in the answer.py file).

## Color Normalization
One way to resolve various lightning conditions in input images is to normalize the color of images. For simplicity, let us use 0.5 as the mean and 0.5 as the standard deviation for each color channel. Run python main.py --model SimpleCNN --transform norm.

## Deep Convolutional Neural Network
CNNs with only one convolutional layer can only extract simple features from images, but deeper CNNs can extract more complex information.

As shown in the following table, when given an array (e.g. [8, 16, 32, "pool"]), the DeepCNN should create a deep network with corresponding convolutional layers (i.e. 8-channel convLayer, 16-channel convLayer, 32-channel convLayer, max pooling layer), and then add a fully-connected layer after the last convolutional (or pooling) layer.

<img width="200" alt="image" src="https://user-images.githubusercontent.com/74432509/222872714-5eb79a6b-3507-4259-b34d-bad226482b0b.png">

I assume the input data for this model is always a 3 × 30 × 30 PyTorch Tensor (which is a 30 × 30 RGB image). I always use 2D convolutional layers with kernel size 3, stride 1, padding 0, dilation 1, and group 1; I also add a ReLU activation function after every convolutional layer. Similarly, we will always use max-pooling layers with kernel size 2, padding 0, and dilation 1.

I had to reshape tensors to (b × p) size before feeding them to the first fully-connected layer, where b is the batch size and p is the length of feature vector.

With a 3 hidden-layer CNN (i.e. [8, 16, 32, "pool"]), run python main.py -m DeepCNN --layers 8 16 32 pool --transform norm.
### (Code coming soon)
