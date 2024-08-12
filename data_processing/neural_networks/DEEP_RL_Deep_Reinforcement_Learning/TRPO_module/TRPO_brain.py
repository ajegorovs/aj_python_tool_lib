import torch, torch.nn as nn
from torch.optim import Adam
from typing import List
from data_processing.neural_networks.DEEP_RL_Deep_Reinforcement_Learning.TRPO_module.TRPO_MLP import torch_MLP,torch_MLP_Normal
from torch.nn.utils import parameters_to_vector         # https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical # https://pytorch.org/docs/stable/distributions.html#categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

# ============================== Critic - value function ==============================
class value(torch_MLP):
    def __init__(self, inp_size, out_size, hidden_sizes, activation=nn.ReLU, activation_out=nn.Identity, optimizer= Adam, lr=0.01) -> None:
        super().__init__(inp_size, out_size, hidden_sizes, activation, activation_out, optimizer, lr)
        self.loss = nn.MSELoss()
    
    def train(self, observations, rewards, n_iters = 50):
        for _ in range(n_iters):
            self.optimizer.zero_grad()
            x = self(observations)
            y = rewards.reshape(x.shape)
            output = self.loss(x,y)
            output.backward()
            self.optimizer.step()

# ============================== Actor - policy ==============================
# ========= Parent policy to discrete and continuous action spaces ===========

class TRPO_Proto_Policy():
    """all TRPO policies will have: 1) NN, 2) gradient, 3) log_p_old"""
    def __init__(self, obs_dim: int, num_actions:int, hidden_sizes:List[int], model = torch_MLP, activation=nn.Tanh, activation_out=nn.Identity, optimizer = None, lr: float=0.01) -> None:
        self.mlp = model(obs_dim, num_actions, hidden_sizes, activation, activation_out, optimizer, lr)
        self.tensor_params = self.mlp.tensor_params
        self.D_KL           = torch.tensor(0.0, **self.tensor_params)
        self.U_rew          = torch.tensor(0.0, **self.tensor_params)
        self.U_rew_old      = torch.tensor(0.0, **self.tensor_params)
        self.MLP_num_prms   = sum(p.numel() for p in self.mlp.parameters())
        self.grad           = torch.zeros(size = (self.MLP_num_prms ,), **self.tensor_params)
        self.logP_old= []
        self.logP    = []
        self.logP_grads     = []

    def get_policy(self, observation): 
        """Returns a distribution of probabilities for a particular observation/-s."""
        pass

    def get_action(self, observation):
        """Samples action from distribution. Remembers log-prob of given action."""
        distribution    = self.get_policy(observation)
        action          = distribution.sample()
        return action
    
    def batch_calc_logPs(self, observations, actions):
        """ Generate log-probs using current version of policy"""
        distribution    = self.get_policy(observations)
        self.logP       = distribution.log_prob(actions).flatten()  # shape: (batch,)
        return distribution

    def prep_old_policy(self):
        self.logP_old = self.logP.clone().detach()

    def calc_KL_div(self):
        """Calculate KL divergence. Implementation depends on action space type"""
        pass

    def calc_flat_grad(self, f, create_graph = False, retain_graph = True):
        """ Calculate gradient of scalar function with respect to network parameters.
            create_graph=True for higher order derivatives, retain_graph = False will 
            wipe intermediate values from graph. Its a big no-no"""
        # Retain by default, and by default no higher order grads.
        g = torch.autograd.grad(f, self.mlp.parameters(), create_graph = create_graph, retain_graph = retain_graph)
        return parameters_to_vector(g)
    
    def calc_NPG_logP_grads(self):
        """ Calculates gradient for each 
            log-probability of action a taken in state s -> grad(log(a_t|s_t))
            for all t in a batch. These are needed for NPG grad and FIM"""
        self.logP_grads = torch.zeros(size = (len(self.logP),self.MLP_num_prms), **self.tensor_params)
        for i in range(len(self.logP)):
            self.logP_grads[i] = self.calc_flat_grad(self.logP[i])
        return self.logP_grads 
    
    def calc_NPG_grad(self, advantages):
        """Calculates NPG type gradient of objective function"""
        grad = self.logP_grads*advantages.reshape(-1,1)
        grad = grad.mean(dim=0)
        return grad
    
    def calc_surrogate_reward(self, advantages):
        """Calculate TRPO type objective function /w importance sampling"""
        loss = torch.exp(self.logP - self.logP_old)*(advantages.flatten())
        loss = loss.mean(dim = 0)
        return loss
    
    def calf_FIM(self, regularization = 0.001):
        """ Calculates Fisher's Information Matrix (FIM) via gradients of logP_old.
            Relevant only for old policy to get curvature of parameter space"""
        FIM = self.logP_grads.unsqueeze(-1) * self.logP_grads.unsqueeze(-2) # batched outer product via broadcasting (T,N,1) * (T,1,N)
        FIM = FIM.mean(dim=0) 
        FIM += regularization * torch.eye(self.MLP_num_prms, device=device) 
        return FIM  
    
    def calc_Hessian(self, regularization = 0.001):
        """"Calculate Hessian as the Jacobian of a gradient.
            1) Create gradient of KL divergence (with graph enabled);
            2) Go through each entry and calculate gradient;
            3) drop from graph since we need to know direction once."""                     
        flat_KL_grad = self.calc_flat_grad(self.D_KL, create_graph=True)
        H = torch.zeros(size = (self.MLP_num_prms,)*2, **self.tensor_params)
        for i in range(len(flat_KL_grad)):
            H[i] = self.calc_flat_grad(flat_KL_grad[i])
        H.requires_grad_(False)
        H += regularization * torch.eye(self.MLP_num_prms, **self.tensor_params)
        return H
    
# ========= Child policies for discrete and continuous action spaces ===========
class Policy_Discrete(TRPO_Proto_Policy):
    """"""
    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: List[int], model=torch_MLP, activation=nn.Tanh, activation_out=nn.Identity, optimizer=None, lr: float = 0.01) -> None:
        super().__init__(obs_dim, num_actions, hidden_sizes, model, activation, activation_out, optimizer, lr)
        """For discrete case need (and can access) to probability distributions over actions for any state"""
        self.logP_distr       = []  #batch length x num actions
        self.logP_distr_old   = []  

    def get_policy(self, observation): 
        """Distribution for discrete actions"""
        return Categorical(logits=self.mlp(observation))
    
    def get_action(self, observation):
        """Added saving of log P(.,s) for D_KL"""
        return super().get_action(observation).item()
   
    def batch_calc_logPs(self, observations, actions):
        """ For discrete case calculate:
            -   for importance sampling: (log) probabilities for chosen action for each state in trajectory
            -   for KL divergence prob distr over actions (for all visited states St at time t)
                distr(St) = log [P(a1,St), P(a2,St), ...]"""
        distribution = super().batch_calc_logPs(observations, actions.flatten())
        self.logP_distr   = torch.log(distribution.probs)
        
    def prep_old_policy(self):
        """Create old version of policy without gradient enabled"""
        super().prep_old_policy()
        self.logP_distr_old   = self.logP_distr.clone().detach()

    def calc_KL_div(self):
        """ Defined KL divergence specifically for discrete actions. 
            NOTE: logP_all uses grad, logP_all_old does not."""
        temp = torch.exp(self.logP_distr_old)*(self.logP_distr_old- self.logP_distr)
        return temp.sum(dim=-1).mean()

class Policy_continuous(TRPO_Proto_Policy):
    """ NN produces mean value vector parameter for Gaussian distribution, from which actions are sampled.
        state maps to mean vector, Gaussian distribution, additional trainable parameters are stdev 
        for all actions.
        for KL divergence we collect mean and stdev for each visited state."""
    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: List[int], model=torch_MLP_Normal, activation=nn.Tanh, activation_out=nn.Identity, optimizer=None, lr: float = 0.01) -> None:
        super().__init__(obs_dim, num_actions, hidden_sizes, model, activation, activation_out, optimizer, lr)
        self.mus , self.mus_old = [], []
        self.log_std_old        = torch.zeros(size=(num_actions,), **self.tensor_params)

    def calc_log_prob(self, action, mu, log_std):
        """NOT USED: can be calculated using Normal(mu,std).log_prob(action)"""
        return -0.5*(torch.log(2*torch.tensor(torch.pi))+ 2*log_std + ((action - mu)/(torch.exp(log_std)+ EPS))**2)
    
    def get_policy(self, observation):
        """Get Multivariate Normal distribution"""
        mus  = self.mlp(observation)
        stds = self.mlp.log_std.exp()
        return MultivariateNormal(mus, torch.diag(stds))

    def get_action(self, observation):
        """Sample from gaussian, remember parameter"""
        return super().get_action(observation).cpu().numpy() # array, not int
    
    def batch_calc_logPs(self, observations, actions):
        """ For contious case calculate:
            -   for importance sampling: (log) probabilities for chosen action for each state in trajectory
            -   for KL divergence parameters of multivariate gaussian (for all visited states St at time t)
                [mu(S0), mu(S1), ...]; state does not determine stdev."""
        distribution = super().batch_calc_logPs(observations, actions)
        self.mus = distribution.mean

    def prep_old_policy(self):
        """Create old version of policy without gradient enabled"""
        super().prep_old_policy()
        self.mus_old        = self.mus.clone().detach()
        self.log_std_old    = self.mlp.log_std.clone().detach()

    def KL_div(self, mu_old, log_std_old, mu, log_std):
        var_old, var = torch.exp(2*log_std_old), torch.exp(2*log_std)
        dkl = 0.5*(((mu_old - mu)**2 + var_old)/var - 1) + log_std - log_std_old
        return dkl.sum(dim = -1).mean()
    
    def calc_KL_div(self):
        mu_old, log_std_old = self.mus_old  , self.log_std_old
        mu, log_std         = self.mus      , self.mlp.log_std
        return self.KL_div(mu_old, log_std_old, mu, log_std)