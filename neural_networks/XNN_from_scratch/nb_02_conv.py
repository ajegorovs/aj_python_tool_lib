import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor

from network import train, predict
from layer_conv import Convolution
from activation_functions import Sigmoid
from layer_reshape import Reshape
from layer_dense import Dense
from loss_functions import (binary_cross_entropy, binary_cross_entropy_prime)
subset = [0,1,2]
target_transform = lambda y: torch.zeros(len(subset), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1).numpy()

def pre_proc(subset, train = True, max_vals = 100):
    data_train = iter(torchvision.datasets.MNIST(r'../microsoft_intro-machine-learning-pytorch/data', download=True,train=train,transform=ToTensor()))
    asd = [];i = 0
    for k in range(1000):
        (a,b) = next(data_train)
        if b in subset:
            i += 1
            asd.append((a,b))
        if i >= max_vals:
            break
    data_train = asd
    data    = np.array([a[0] for a in data_train])
    print(np.min(data),np.max(data))
    #data    /= 255.0;print(np.max(data))
    labels  = np.array([target_transform(a[1]) for a in data_train]).reshape(-1,len(subset),1)

    return data, labels

x_train , y_train   = pre_proc(subset, train = True)
if 1 == -1:
    fig,ax = plt.subplots(1,7)
    for i in range(7):
        ax[i].imshow(x_train[i][0], cmap='gray')
        ax[i].set_title(y_train[i])
        ax[i].axis('off')
    plt.show()


 # neural network
network = [
    Convolution((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, len(subset)),
    Sigmoid()
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=40,
    learning_rate=0.1
)

# test
x_test  , y_test    = pre_proc(subset, train = False)
success = np.zeros(y_test.shape[0])
i = 0
for x, y in zip(x_test, y_test):

    output = predict(network, x)
    aa = np.argmax(output)
    #print(f"pred: {aa}:{output, output[aa]}, true: {np.argmax(y)}")
    success[i] = np.argmax(output) == np.argmax(y)
    #print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    i += 1
    
print('success:', np.mean(success))