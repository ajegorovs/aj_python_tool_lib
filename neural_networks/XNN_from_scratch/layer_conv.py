from c_layer import Layer
import numpy as np
from scipy import signal

class Convolution(Layer):
    """
    Take an input, say RGB image of HxW dims /w 3 channels.
    Apply convolve with 3 channel kernel, and sum into one layer.
    Produced layer is not padded. so its smaller. add bias.
    More kernels can be used and multiple outputs obtained
    """
    def __init__(self, channels, height, width, kernels_width, kernels_num):
        self.input_num_channels     = channels
        self.input_shape    = (channels, height, width)
         
        self.output_num   = kernels_num
        # multiple kernels with same channels as input data
        self.kernel_shape = (kernels_num, self.input_num_channels, kernels_width, kernels_width)
        # channels are combined after convolve, no padding = smaller target output
        self.output_shape = (kernels_num, height - kernels_width + 1, width - kernels_width + 1)
        # predefine arrays in memory. kernels and bias will be changed via backwards method.
        self.kernels    = np.random.randn(*self.kernel_shape)
        self.bias       = np.random.randn(*self.output_shape)
        # output is a temp storage for intermediate stages for summing channels.
        self.output     = np.random.randn(*self.output_shape)
        # reserve memory for gradients. kernel for gradient discent, input for return.
        self.input_grad     = np.zeros(self.input_shape)
        self.kernels_grad   = np.zeros_like(self.kernels).astype(float)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias)                # virtually reset output and add bias.

        for i in range(self.output_num):                # number of 'RGB' kernels applied. means 1 kernel has '3' channels.
            for j in range(self.input_num_channels):    # apply each channel of each kernel to input channels. 
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], mode='valid') # i-th kern, j-th chann

        return self.output
    
    def backwards(self, output_gradient, learning_rate):
        self.input_grad     *= 0.0
        self.kernels_grad   *= 0.0
        for i in range(self.output_num):
            for j in range(self.input_num_channels):
                self.kernels_grad[i,j]  = signal.correlate2d(self.input[j]      , output_gradient[i], mode='valid')
                self.input_grad[j]      += signal.convolve2d(output_gradient[i] , self.kernels[i, j], mode='full' )
                
        self.bias       -= learning_rate * output_gradient
        self.kernels    -= learning_rate * self.kernels_grad

        
        return self.input_grad
    

class Convolution2(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
