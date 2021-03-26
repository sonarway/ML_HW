import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1 )
        self.relu1 = ReLULayer()
        self.max_pool1 = MaxPoolingLayer(4,4)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu2 = ReLULayer()
        self.max_pool2 = MaxPoolingLayer(4,4)
        self.flat = Flattener()
        self.fc = FullyConnectedLayer(4*conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        
        preds = self.max_pool1.forward(self.relu1.forward(self.conv1.forward(X)))
        preds = self.max_pool2.forward(self.relu2.forward(self.conv2.forward(preds)))
        preds = self.flat.forward(preds)
        preds = self.fc.forward(preds)

        loss, grad = softmax_with_cross_entropy(preds, y)

        grad = self.flat.backward(self.fc.backward(grad))
        grad = self.conv2.backward(self.relu2.backward(self.max_pool2.backward(grad)))
        grad = self.conv1.backward(self.relu1.backward(self.max_pool1.backward(grad)))

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        step1 = self.max_pool1.forward(self.relu1.forward(self.conv1.forward(X)))
        step2 = self.max_pool2.forward(self.relu2.forward(self.conv2.forward(step1)))
        step3 = self.flat.forward(step2)
        step4 = self.fc.forward(step3)
        probs = softmax(step4)
        pred = np.argmax(probs, axis=1)

        return pred


    def params(self):

        # TODO: Aggregate all the params from all the layers
        # which have parameters

        return {'W1': self.conv1.W, 'B1': self.conv2.B,
        'W2': self.fc.W, 'B2': self.fc.B,
        'W3': self.conv2.W, 'B3': self.conv2.B,
        }
