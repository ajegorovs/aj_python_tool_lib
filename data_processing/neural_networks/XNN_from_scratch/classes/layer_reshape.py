from classes.c_layer import Layer
import numpy as np

class Reshape(Layer):
    def __init__(self, shape_input, shape_output):
        self.shape_input = shape_input
        self.shape_output = shape_output

    def forward(self, input):
        return np.reshape(input, self.shape_output)
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.shape_input)