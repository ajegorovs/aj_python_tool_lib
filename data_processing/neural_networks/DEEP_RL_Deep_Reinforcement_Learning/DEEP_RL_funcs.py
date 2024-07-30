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
    def __init__(self, env_name, device = 'cpu', dtype = float, *args) -> None:
        self.env = gym.make(env_name, *args)
        self.batch_states       = []
        self.batch_actions      = []
        self.batch_rewards      = []
        self.episode_rewards    = []
        self.env_iters          = 0
        self.device             = device
        self.dtype              = float

    def clear_batches(self):
        self.batch_states  = []
        self.batch_actions = []
        self.batch_rewards = []

    def reset(self):
        state = self.env.reset()[0]
        state = torch.as_tensor(state, device=self.device, dtype = float)
        self.batch_states.append(state)
        self.episode_rewards    = []
        return state
    
    def step(self, action):
        self.batch_actions.append(action)
        state, reward, done = self.env.step(action)[:3]
        state = torch.as_tensor(state, device=self.device, dtype = self.dtype)
        self.episode_rewards.append(reward)
        return state, reward, done
    
    def add_ep_rewards_2_go(self):
        #self.batch_rewards      += list(np.flip(np.cumsum(np.flip(self.episode_rewards))))
        ER = torch.tensor(self.episode_rewards, dtype= self.dtype)
        self.batch_rewards      += list(torch.flip(torch.cumsum(torch.flip(ER,dims=(0,)),dim=0),dims=(0,)))

    def add_ep_rewards_cum(self):
        self.batch_rewards      += [self.dtype(sum(self.episode_rewards))]*len(self.episode_rewards)

    @torch.no_grad()
    def play_batch(self, policy, max_batch_size, max_ep_len = 300, rew_type = None, tqdm_iter_params = None):
        # switch future reward approach. r2g = rewards to go. else cum-sum
        calc_rew = self.add_ep_rewards_2_go if rew_type in [None, 'r2g'] else self.add_ep_rewards_cum
        # reporting via tqdm
        if tqdm_iter_params is not None:
            tqdm_iter, tqdm_params = tqdm_iter_params
        # init
        self.clear_batches()
        obs         = self.reset()
        t           = 0
        batch_times = []
        batch_L             = 0
        while True:
            if tqdm_iter_params is not None:
                tqdm_iter.set_postfix(**tqdm_params, batch_size = batch_L)
            action                      = policy.get_action(obs)
            obs, _, episode_finished    = self.step(action)
            t                           += 1
            if episode_finished or t >= max_ep_len:
                self.env_iters += 1
                batch_times.append(t)

                calc_rew()
                t = 0
                batch_L = len(self.batch_states)
                if batch_L >= max_batch_size: 
                    break               # end of batch. dont add terminal obs to batch_state
                obs = self.reset()    # start new episode. adds obs to batch_state
            else:
                self.batch_states.append(obs)

        self.batch_states     = torch.stack(self.batch_states  ).to(self.device).to(self.dtype)
        self.batch_rewards    = torch.tensor(self.batch_rewards).to(self.device).to(self.dtype)
        self.batch_actions    = torch.tensor(self.batch_actions).to(self.device).to(self.dtype)

        return torch.as_tensor(batch_times).to(float)
    
class MLP():
    def __init__(self, inp_size, out_size, hidden_sizes, activation = nn.Tanh,activation_out = nn.Identity, lr = 1e-2) -> None:
        self.dtype          = float
        self.act            = activation
        self.act_out        = activation_out
        self.layer_sizes    = [inp_size] + hidden_sizes + [out_size]
        self.mlp            = self.seq()
        self.optimizer      = Adam(self.mlp.parameters(), lr=lr)
        self.device         = "cpu"

    def seq(self):
        layers      = []
        for i,j in zip(self.layer_sizes[:-1],self.layer_sizes[1:]):
            layers  += [nn.Linear(i,j, dtype=self.dtype), self.act()]
        layers[-1]  = self.act_out()
        return nn.Sequential(*layers)
    
    def to(self, device):
        self.mlp.to(device)
        self.device = device  
    
class policy(MLP):
    def __init__(self, inp_size, out_size, hidden_sizes, activation=nn.Tanh, activation_out=nn.Identity, lr=0.01) -> None:
        super().__init__(inp_size, out_size, hidden_sizes, activation, activation_out, lr)

    def get_policy(self,observation):
        return Categorical(logits=self.mlp(observation))
    
    def get_action(self, observation):
        return self.get_policy(observation).sample().item()
    
    def log_prob(self, observations, actions):
        return self.get_policy(observations).log_prob(torch.as_tensor(actions, device= self.device)).to(self.dtype)
  

class value(MLP):
    def __init__(self, inp_size, out_size, hidden_sizes, activation=nn.Tanh, activation_out=nn.Identity, lr=0.01) -> None:
        super().__init__(inp_size, out_size, hidden_sizes, activation, activation_out, lr)
        self.loss = nn.MSELoss()

    def get_value(self, observation):
        return self.mlp(observation).to(self.dtype)
    
    def train(self, observations, rewards, n_iters = 50):
        
        for _ in range(n_iters):
            self.optimizer.zero_grad()
            x = self.get_value(observations)
            y = rewards.reshape(x.shape)
            output = self.loss(x,y)
            output.backward()
            self.optimizer.step()