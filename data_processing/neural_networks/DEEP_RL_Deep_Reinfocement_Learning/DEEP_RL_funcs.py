import torch, gym, numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
#https://pytorch.org/docs/stable/distributions.html
#https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class env_info():
    """this 'wrapper' modifies env control
        automatically adds non terminal states, actions and reward into a buffer
        tracks episode rewards and calculates cumulative future rewards
        """
    def __init__(self, env_name, *args) -> None:
        self.env = gym.make(env_name, *args)
        self.batch_states       = []
        self.batch_actions      = []
        self.batch_rewards      = []
        self.episode_rewards    = []
        self.env_iters          = 0

    def clear_batches(self):
        self.batch_states  = []
        self.batch_actions = []
        self.batch_rewards = []

    def reset(self):
        state = self.env.reset()[0]
        self.batch_states.append(state)
        self.episode_rewards    = []

        return state
    
    def step(self, action):
        self.batch_actions.append(action)
        state, reward, done = self.env.step(action)[:3]
        self.episode_rewards.append(reward)
        return state, reward, done
    
    def add_ep_rewards_2_go(self):
        self.batch_rewards      += list(np.flip(np.cumsum(np.flip(self.episode_rewards))))

    def add_ep_rewards_cum(self):
        self.batch_rewards      += [sum(self.episode_rewards)]*len(self.episode_rewards)
    
class MLP():
    def __init__(self, inp_size, out_size, hidden_sizes, activation = nn.Tanh,activation_out = nn.Identity, lr = 1e-2) -> None:
        self.dtype          = float
        self.act            = activation
        self.act_out        = activation_out
        self.layer_sizes    = [inp_size] + hidden_sizes + [out_size]
        self.mlp            = self.seq()
        self.optimizer      = Adam(self.mlp.parameters(), lr=lr)

    def seq(self):
        layers      = []
        for i,j in zip(self.layer_sizes[:-1],self.layer_sizes[1:]):
            layers  += [nn.Linear(i,j, dtype=self.dtype), self.act()]
        layers[-1]  = self.act_out()
        return nn.Sequential(*layers)
    
class policy(MLP):
    def __init__(self, inp_size, out_size, hidden_sizes, activation=nn.Tanh, activation_out=nn.Identity, lr=0.01) -> None:
        super().__init__(inp_size, out_size, hidden_sizes, activation, activation_out, lr)

    def get_policy(self,observation):
        return Categorical(logits=self.mlp(torch.as_tensor(observation, dtype= self.dtype)))
    
    def get_action(self, observation):
        return self.get_policy(observation).sample().item()
    
    def log_prob(self, observations, actions):
        return self.get_policy(observations).log_prob(torch.as_tensor(actions))
  

class value(MLP):
    def __init__(self, inp_size, out_size, hidden_sizes, activation=nn.Tanh, activation_out=nn.Identity, lr=0.01) -> None:
        super().__init__(inp_size, out_size, hidden_sizes, activation, activation_out, lr)

    def get_value(self, observation):
        return self.mlp(torch.as_tensor(observation, dtype= self.dtype))
    
    def train(self, observations, rewards, n_iters = 50):
        loss = nn.MSELoss()
        for _ in range(n_iters):
            self.optimizer.zero_grad()
            x = self.get_value(observations)
            y = torch.as_tensor(rewards, dtype= self.dtype).reshape(x.shape)
            output = loss(x,y)
            output.backward()
            self.optimizer.step()