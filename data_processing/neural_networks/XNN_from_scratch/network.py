def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

import matplotlib.pyplot as plt
import numpy as np


def update_plot(x_data, y_data, plt_range_x, plt_range_y):
    plt.clf()
    plt.plot(x_data, y_data)#, label='Updated Data')
    #plt.yscale("log")
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.xlim(*plt_range_x)
    plt.ylim(*plt_range_y)
    #plt.legend()
    plt.draw()
    plt.pause(0.001)

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    if verbose:
        plt_x = np.arange(0, epochs).astype(float)
        plt_y = np.zeros_like(plt_x)
        z_len = len(str(epochs))

    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            plt_y[e] = error
            if e % 5 == 0: update_plot(plt_x[:e], plt_y[:e], (0, epochs), (0, plt_y[0]))
            print(f"{e + 1:>{z_len}}/{epochs}, error={error}")

