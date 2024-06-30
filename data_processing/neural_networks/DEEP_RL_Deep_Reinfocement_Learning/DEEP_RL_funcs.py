import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
#https://pytorch.org/docs/stable/distributions.html
#https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py

class MLP():
    def __init__(self, inp_size, out_size, hidden_sizes, activation = nn.Tanh,activation_out = nn.Identity) -> None:
        self.act = activation
        self.act_out = activation_out
        self.layer_sizes = [inp_size] + hidden_sizes + [out_size]
        self.mlp = self.seq()

    def seq(self):
        layers = []
        for i,j in zip(self.layer_sizes[:-1],self.layer_sizes[1:]):
            layers += [nn.Linear(i,j), self.act()]

        layers[-1] = self.act_out()

        return nn.Sequential(*layers)

class Simple_PG(MLP):
    def __init__(self, inp_size, out_size, hidden_sizes, activation=nn.Tanh, activation_out=nn.Identity) -> None:
        super().__init__(inp_size, out_size, hidden_sizes, activation, activation_out)
        self.env_iters = 0

    def get_policy(self,observation):
        return Categorical(logits=self.mlp(torch.as_tensor(observation)))
    
    #@torch.no_grad()
    def get_action(self, observation):
        #observation = torch.Tensor(observation)
        return self.get_policy(observation).sample().item()
    
    #@torch.no_grad()
    def log_prob(self, observations, actions):
        return self.get_policy(torch.as_tensor(observations)).log_prob(torch.as_tensor(actions))
    