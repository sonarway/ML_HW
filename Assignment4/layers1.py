import numpy as np


# reg_strength - λ
def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      :param W, np array - weights
      :param reg_strength - float value
    Returns:
      :returns loss, single value - l2 regularization loss
      :returns gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    loss = reg_strength * np.sum(W ** 2)   # L2(W) = λ * tr(W.T * W)
    grad = 2 * reg_strength * W                         # dL2(W)/dW = 2 * λ * W

    return loss, grad   # L2(W), dL2(W)/dW


# predictions - Z
def softmax(predictions):
    """
    Computes probabilities from scores
    Arguments:
      :param predictions, np array, shape is either (N) or (batch_size, N) - classifier output
    Returns:
      :returns probs, np array of the same shape as predictions - probability for every class, 0..1
    """

    max_pred = np.max(predictions, axis=1, keepdims=True)
    return np.exp(predictions - max_pred) / np.sum(np.exp(predictions - max_pred), axis=-1, keepdims=True)


# probs - S
# target_index - y
def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss
    Arguments:
      :param probs, np array, shape is either (N) or (batch_size, N) - probabilities for every class
      :param target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)
    Returns:
      loss: single value
    """

    rows = np.arange(target_index.shape[0])
    cols = target_index

    return np.mean(-np.log(probs[rows, cols]))  # L


# predictions - Z
# target_index - y
def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions, including the gradient
    Arguments:
      :param predictions, np array, shape is either (N) or (batch_size, N) - classifier output
      :param target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)
    Returns:
      :returns loss, single value - cross-entropy loss
      :returns dprediction, np array same shape as predictions - gradient of predictions by loss value
    """

    probs = softmax(predictions)                    # S
    loss = cross_entropy_loss(probs, target_index)  # L

    indicator = np.zeros(probs.shape)
    indicator[np.arange(probs.shape[0]), target_index] = 1      # 1(y)
    dprediction = (probs - indicator) / predictions.shape[0]    # dL/dZ = (S - 1(y)) / N

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

    def reset_grad(self):
        self.grad = np.zeros_like(self.value)


        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.diff = (X > 0).astype(float)  # dZ/dX
        return np.maximum(X, 0)                             # Z

    # d_out - dL/dZ
    def backward(self, d_out):
        """
        Backward pass
        Arguments:
          :param d_out, np array (batch_size, num_features) - gradient of loss function with respect to output
        Returns:
          :returns d_result: np array (batch_size, num_features) - gradient with respect to input
        """
        d_result = np.multiply(d_out, self.diff)    # dL/dX = dL/dZ * dZ/dX

        return d_result     # dL/dX

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    # d_out - dL/dZ
    def backward(self, d_out):
        """
        Backward pass:
        computes gradient with respect to input and accumulates gradients within self.W and self.B
        Arguments:
          :param d_out, np array (batch_size, n_output) - gradient of loss function with respect to output
        Returns:
          :returns d_result: np array (batch_size, n_input) - gradient with respect to input
        """

        d_result = np.dot(d_out, self.W.value.T)     # dL/dX = dL/dZ * dZ/dX = dL/dZ * W.T                
        self.W.grad += np.dot(self.X.T, d_out) # dL/dW = dL/dZ * dZ/dW = X.T * dL/dZ
        self.B.grad += 2 * np.mean(d_out, axis=0) # dL/dW = dL/dZ * dZ/dB = I * dL/dZ

        return d_result     # dL/dX

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, filter_size, padding):
        """
        Initializes the layer
        Arguments:
          :param in_channels, int - number of input channels
          :param out_channels, int - number of output channels
          :param filter_size, int - size of the conv filter
          :param padding, int - number of 'pixels' to pad on each side
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.padding = padding

        self.W = Param(np.random.randn(filter_size, filter_size, in_channels, out_channels))
        self.B = Param(np.zeros(out_channels))

        self.X_pad = None

    def forward(self, X):
        padding = self.padding
        batch_size, in_height, in_width, in_channels = X.shape

        in_height_pad = in_height + 2 * padding
        in_width_pad = in_width + 2 * padding

        self.X_pad = np.zeros((batch_size, in_height_pad, in_width_pad, in_channels))
        self.X_pad[:, padding:in_height + padding, padding:in_width + padding, :] = X

        batch_size, in_height, in_width, in_channels = self.X_pad.shape
        assert in_channels == self.in_channels

        filter_size = self.filter_size
        out_channels = self.out_channels

        out_height = in_height - filter_size + 1
        out_width = in_width - filter_size + 1

        Z = np.zeros((batch_size, out_height, out_width, out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                X_window = self.X_pad[:, y:y+filter_size, x:x+filter_size, :]   # batch @ filter @ filter @ in_channels
                X_window_2d = X_window.reshape(batch_size, -1)                  # batch @ (filter * filter * in_channels)
                W_2d = self.W.value.reshape(-1, out_channels)                   # (filter * filter * in_channels) @ out_channels
                # batch @ out_channels
                Z[:, y, x, :] = np.dot(X_window_2d, W_2d) + self.B.value

        return Z

    # d_out - dL/dZ
    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        in_batch_size, in_height, in_width, in_channels = self.X_pad.shape
        out_batch_size, out_height, out_width, out_channels = d_out.shape

        assert in_batch_size == out_batch_size
        assert in_channels == self.in_channels
        assert out_channels == self.out_channels

        assert out_height == in_height - self.filter_size + 1
        assert out_width == in_width - self.filter_size + 1

        batch_size = in_batch_size
        filter_size = self.filter_size

        d_result = np.zeros((batch_size, in_height, in_width, in_channels))

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                X_window = self.X_pad[:, y:y + filter_size, x:x + filter_size, :]           # batch @ filter @ filter @ in_channels

                X_window_2d = X_window.reshape(batch_size, -1)                              # batch @ (filter * filter * in_channels)
                W_2d = self.W.value.reshape(-1, self.out_channels)                          # (filter * filter * in_channels) @ out_channels
                d_out_2d = d_out[:, y, x, :]                                                # batch @ out_channels

                d_result_window_2d = np.dot(d_out_2d, W_2d.T)                            # batch @ (filter * filter * in_channels)
                dLdW_2d = np.dot(X_window_2d.T, d_out_2d)                                # (filter * filter * in_channels) @ out_channels
                dLdB = 2 * np.mean(d_out_2d, axis=0)                                        # out_channels

                d_result_window_shape = (batch_size, filter_size, filter_size, in_channels)
                d_result_window = d_result_window_2d.reshape(d_result_window_shape)         # batch @ filter @ filter @ in_channels

                dLdW_shape = (filter_size, filter_size, in_channels, out_channels)
                dLdW = dLdW_2d.reshape(dLdW_shape)                                          # filter @ filter @ in_channels @ out_channels

                d_result[:, y:y + filter_size, x:x + filter_size, :] += d_result_window
                self.W.grad += dLdW
                self.B.grad += dLdB

        padding = self.padding
        batch_size, in_height_pad, in_width_pad, in_channels = d_result.shape

        in_height = in_height_pad - 2 * padding
        in_width = in_width_pad - 2 * padding

        return d_result[:, padding:in_height + padding, padding:in_width + padding, :]

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        """
        Initializes the max pool
        Arguments:
          :param pool_size, int - area to pool
          :param stride, int - step size between pooling windows
        """
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        batch_size, in_height, in_width, channels = X.shape

        out_height = int((in_height - self.pool_size) / self.stride) + 1
        out_width = int((in_width - self.pool_size) / self.stride) + 1

        M = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):

                pool_y_from = y * self.stride
                pool_y_to = pool_y_from + self.pool_size
                pool_x_from = x * self.stride
                pool_x_to = pool_x_from + self.pool_size

                M[:, y, x, :] = np.amax(X[:, pool_y_from:pool_y_to, pool_x_from:pool_x_to, :], axis=(1, 2))

        return M

    # d_out - dL/dM
    def backward(self, d_out):
        batch_size, in_height, in_width, channels = self.X.shape

        out_height = int((in_height - self.pool_size) / self.stride) + 1
        out_width = int((in_width - self.pool_size) / self.stride) + 1

        d_result = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):

                pool_y_from = y * self.stride
                pool_y_to = pool_y_from + self.pool_size
                pool_x_from = x * self.stride
                pool_x_to = pool_x_from + self.pool_size

                # TODO: vectorize
                for b in range(batch_size):
                    for c in range(channels):
                        d_out_pooled = d_out[b, y, x, c]
                        X_pooled = self.X[b, pool_y_from:pool_y_to, pool_x_from:pool_x_to, c]

                        max_ind_y, max_ind_x = np.unravel_index(np.argmax(X_pooled), X_pooled.shape)
                        d_result[b, pool_y_from + max_ind_y, pool_x_from + max_ind_x, c] += d_out_pooled

        return d_result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape

        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}