import numpy as np
from classes.layer_dense import Dense
from classes.activation_functions import Tanh
from classes.loss_functions import (mse, mse_prime)

X = np.array([[0,0],[0,1],[1,0],[1,1]]).reshape(4,2,1)
Y = np.array([[0],[1],[1],[0]]).reshape(4,1,1)

network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]
epochs = 10000
learning_rate = 0.1

for e in range(epochs):
    error = 0

    for (x,y) in zip(X,Y):
        output = x
        for layer in network:
            output = layer.forward(output)

        error += mse(y, output)

        grad = mse_prime(y, output)

        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

        error /= len(X)
    if e % 250 == 0: 
        print('%d/%d error = %f' % (e + 1, epochs, error))