import torch.nn as nn
import torch
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class torch_MLP(nn.Module):
    def __init__(self, inp_size: int, out_size: int, hidden_sizes: List[int],
                 activation = nn.Tanh, activation_out = nn.Identity, 
                 optimizer = None, lr: float = 1e-2) -> None:
        super().__init__()
        self.dtype          = torch.float32
        self.tensor_params  = {'dtype': self.dtype, 'device': device}
        self.act            = activation
        self.act_out        = activation_out
        self.dense          = self.seq([inp_size] + hidden_sizes + [out_size]).to(device)
        self.optimizer = None if optimizer is None else optimizer(self.parameters(), lr=lr)

    def seq(self, layers_size):
        layers      = []
        for i,j in zip(layers_size[:-1],layers_size[1:]):
            layers  += [nn.Linear(i,j, **self.tensor_params), self.act()]
            nn.init.zeros_(layers[-2].weight)  # for logits better to have equal probs
        layers[-1]  = self.act_out()
        return nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.dense(obs)

class torch_MLP_Normal(torch_MLP):
    """ inherits dense layer block, adds extra parameters"""
    def __init__(self, inp_size: int, out_size: int, hidden_sizes:List[int],
                 activation = nn.Tanh, activation_out = nn.Identity, optimizer = None, lr: float = 1e-2) -> None:
        super().__init__(inp_size, out_size, hidden_sizes, activation, activation_out, optimizer, lr)

        self.log_std = nn.Parameter(0.5*torch.ones(size=(out_size,),**self.tensor_params))