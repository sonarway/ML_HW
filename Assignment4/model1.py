import numpy as np

from layers import FullyConnectedLayer, ReLULayer, ConvolutionalLayer, MaxPoolingLayer, Flattener
from layers import softmax, softmax_with_cross_entropy


class ConvNet:
    """
    Implements a very simple conv net
    Input ->
    Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network
        Arguments:
          :param input_shape, tuple of 3 ints - image_width, image_height, n_channels, wll be equal to (32, 32, 3)
          :param n_output_classes, int - number of classes to predict
          :param conv1_channels, int - number of filters in the 1st conv layer
          :param conv2_channels, int - number of filters in the 2nd conv layer
        """
        image_width, image_height, image_channels = input_shape

        maxpool1_size = 4
        maxpool2_size = 4

        flattener_width = int(image_width / (maxpool1_size * maxpool2_size))
        flattener_height = int(image_width / (maxpool1_size * maxpool2_size))

        self.layers = [
            ConvolutionalLayer(in_channels=image_channels, out_channels=conv1_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(maxpool1_size, maxpool1_size),

            ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(maxpool2_size, maxpool2_size),

            Flattener(),
            FullyConnectedLayer(flattener_width * flattener_height * conv2_channels, n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients on a batch of training examples
        Arguments:
          :param X, np array (batch_size, height, width, input_features) - input data
          :param y, np array of int (batch_size) - classes
        """

        assert X.ndim == 4
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        for param in self.params().values():
            param.reset_grad()

        # forward pass
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        # backward pass
        loss, d_out = softmax_with_cross_entropy(out, y)
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

        return loss

    def predict(self, X):
        # forward pass
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        out = softmax(out)

        pred = np.argmax(out, axis=1)
        return pred  # y_hat

    def params(self):
        result = {}
        for index, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                result['%s_%s' % (index, name)] = param

        return result