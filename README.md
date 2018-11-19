# Deep-ResNet-CIFAR100

Built very deep convolutional networks using Residual Networks
(ResNets). In theory, deep networks can represent very complex functions. However, in
practice, they are difficult to train. Residual Networks, introduced by He et al. (2015), allow
for training deeper networks than was previously possible. In this project,
I implemented the basic building blocks of ResNets, and then stacked together these building
blocks to implement and train a deep residual neural network for image classification on
the CIFAR100 dataset via Pytorch. Moreover, in this project, I also learned how
to use a pre-trained ResNet (which was trained on the ImageNet dataset) and then train it
on CIFAR100. This is an example of transfer learning.


• Built the Residual Network and achieved 68% test accuracy.
Defined my own “Basic Block”. For each
weight layer, it contained 3 × 3 filters for a specific number of input channels and
output channels. The output of a sequence of ResNet basic blocks goes through a max
pooling layer with a filter size of 4x4, and then goes to a fully-connected
layer.

• Fine-tuned a pre-trained ResNet-18 model and achieved 77% test accuracy.
