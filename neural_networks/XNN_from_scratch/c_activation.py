from c_layer import Layer
import numpy as np

class Activation(Layer):
    """
    Base class for activation layer. Is a base for specific activation functions.
    Activation functions overwrite forward and backwards functions.
    Backwards method is a derivative of a forward_function.
    """
    def __init__(self, forward_function, backwards_function):
        self.activation = forward_function
        self.activation_prime = backwards_function

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))