import torch, torchvision, matplotlib.pyplot as plt, numpy as np, pickle, os

from torchvision.transforms import ToTensor

from network import train, predict
from classes.layer_conv import Convolution
from classes.activation_functions import Sigmoid
from classes.layer_reshape import Reshape
from classes.layer_dense import Dense
from classes.loss_functions import (binary_cross_entropy, binary_cross_entropy_prime)

subset = [0,1,2]

# create labels 0 = [1,0,0], 1 = [0,1,0], 2 = [0,0,1]
target_transform = lambda y: torch.zeros(len(subset), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1).numpy()
workdir = os.path.split(os.path.relpath(__file__))[0]
os.chdir(workdir)
def pre_proc(subset, train = True, max_vals = 100):
    data_train = iter(torchvision.datasets.MNIST(r'../data', download=True,train=train,transform=ToTensor()))
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

num_kernels = 5
ker_size    = 3
input_shape = np.array(x_train.shape[1:])
H,W = input_shape[1:]
ker_shape   = (ker_size, num_kernels) # num channels not included here. real kernels have shape (num_ker, chan, size, size)
dense_nodes = 100
# colvolved image, w/o padding, has removed border of thickness ker_size//2. Same as H - ker + 1 for odd ker.
convolved_shape = [num_kernels, H - ker_size + 1, W - ker_size + 1]
convolved_shape_flattened = np.prod(convolved_shape)

# store network's parameters in a string. it will be used in a save filename.
# and current network's settings are different it wont load save state
descriptor = f'kn{num_kernels}ks{ker_size}dn{dense_nodes}'

os.mkdir('save_states') if not os.path.exists('save_states') else 0

file_name = os.path.join('save_states','state_'+descriptor + '.pkl')


network = [
    Convolution(*input_shape, *ker_shape),
    Sigmoid(),
    Reshape(convolved_shape, (convolved_shape_flattened, 1)),
    Dense(convolved_shape_flattened, dense_nodes),
    Sigmoid(),
    Dense(dense_nodes, len(subset)),
    Sigmoid()
]
if os.path.exists(file_name) and  1 == 1:
    with open(file_name, 'rb') as file: 
        [
        network[0].kernels , 
        network[0].bias     ,
        network[3].weights  ,
        network[3].bias     ,
        network[5].weights  ,
        network[5].bias     
        ] = pickle.load(file) 

train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

x_test  , y_test    = pre_proc(subset, train = False)
success = np.zeros(y_test.shape[0])
i = 0
for x, y in zip(x_test, y_test):

    output = predict(network, x)
    aa = np.argmax(output)
    #print(f"pred: {aa}:{output, output[aa]}, true: {np.argmax(y)}")
    success[i] = np.argmax(output) == np.argmax(y)
    i += 1
    
print('success:', np.mean(success))


with open(file_name, 'wb') as file: 
      
    # A new file will be created 
    pickle.dump([network[0].kernels, 
                 network[0].bias, 
                 network[3].weights, 
                 network[3].bias, 
                 network[5].weights, 
                 network[5].bias    ], file) 