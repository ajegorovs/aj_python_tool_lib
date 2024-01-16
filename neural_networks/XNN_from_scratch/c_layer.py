class Layer:
    """Base layer class. From it every specialized layer inherits.
    forwards function processes data and passes to next layer.
    backwards fuction passes gradients to prev layer
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass
    