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
            layers += [nn.Linear(i,j), self.act]

        layers[-1] = self.act_out

        return nn.Sequential(*layers)

class policy(MLP):
    def __init__(self, inp_size, out_size, hidden_sizes, activation=nn.Tanh, activation_out=nn.Identity) -> None:
        super().__init__(inp_size, out_size, hidden_sizes, activation, activation_out)
        
    def get_policy(self,observation):
        return Categorical(logits=self.seq(observation))
    
    @torch.no_grad()
    def get_action(self, observation):
        return self.get_policy(observation).sample()
    
    def compute_loss(self,observations, actions, rewards):
        LogPs = self.get_policy(observations).log_prob(actions)
        return -(LogPs * rewards).mean() # -1 to maximize

