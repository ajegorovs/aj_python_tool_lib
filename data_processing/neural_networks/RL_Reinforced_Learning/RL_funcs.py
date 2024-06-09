import numpy as np
from collections import defaultdict
import gym
import matplotlib.pyplot as plt

def update_mean_single(arr_a_mean, arr_a_len, val):
    return 1/(arr_a_len+1) * (arr_a_len*arr_a_mean + val)

class init_env():
    def __init__(self, env_name, *args, eps = 0.1, seed = 69, **kwargs) -> None:
        #np.random.seed(seed)
        self.env                    = gym.make(env_name, *args, **kwargs)
        if type(self.env.observation_space) == gym.spaces.tuple.Tuple:
            self.STATE_SHAPE        = tuple(map(lambda x: x.n, self.env.observation_space))
        else:
            self.STATE_SHAPE        = (self.env.observation_space.n,) 

        self.NUM_ACTIONS            = self.env.action_space.n 
        self.STATE_SHAPE_EXTENDED   = self.STATE_SHAPE + (self.NUM_ACTIONS,)
        self.NUM_STATES_TOTAL       = np.prod(self.STATE_SHAPE)
        self.actions                = np.arange(self.NUM_ACTIONS)
        self.eps                    = eps
        self.policy                 = np.zeros(shape= self.STATE_SHAPE_EXTENDED)
        self.Qsa                    = np.zeros(shape= self.STATE_SHAPE_EXTENDED)
        self.policy_greedy          = np.zeros(shape= self.STATE_SHAPE, dtype = int)
        self.num_state_visits       = defaultdict(int)
        self.test_rewards           = defaultdict(float)
        self.steps_per_ep           = defaultdict(float)
        self.Csa                    = defaultdict(float)
        self.env_iters              = 0
            
    def get_policy_soft(self, policy=None, random = False, action = 0, eps = None):
        if eps is None: eps = self.eps
        if policy is None: policy = self.policy
        # define base epsilon-soft policy. 1) fill with eps values;2) fill greedy values
        policy      += eps/(self.NUM_ACTIONS)
        policy      = policy.reshape((-1,self.NUM_ACTIONS))
        num_states  = len(policy)
        if random:  greedy  = np.random.randint(0, self.NUM_ACTIONS, size = (num_states,))
        else:       greedy  = np.array([action]*num_states)
        policy[np.arange(self.NUM_STATES_TOTAL),greedy] = (1 - eps + eps/(self.NUM_ACTIONS))
        policy = policy.reshape(self.STATE_SHAPE_EXTENDED)
        return policy
    
    def get_Qsa(self, Qsa = None, random = False, action = None, val = 0.5):
        if Qsa is None: Qsa = self.Qsa
        if not random:
            if action is not None:  Qsa[...,action] += val
            else:                   Qsa             += val
        else:
            Qsa += np.random.randn(*Qsa.shape) # need in-place, in case i pass non None Qsa

        return Qsa
    
    def get_policy_Qsa_soft(self, policy = None, Qsa = None, eps = None):
        if policy   is None: policy = self.policy
        if Qsa      is None: Qsa    = self.Qsa
        if eps      is None: eps    = self.eps
        greedy =  np.argmax(Qsa, axis=-1).flatten()
        policy =  policy.reshape((-1,self.NUM_ACTIONS))
        policy *= 0
        policy += eps/self.NUM_ACTIONS
        policy[np.arange(self.NUM_STATES_TOTAL),greedy] = (1 - eps + eps/(self.NUM_ACTIONS))
        policy = policy.reshape(self.STATE_SHAPE_EXTENDED)
        return policy
    
    def get_policy_Qsa_greedy(self):
        self.policy_greedy = np.argmax(self.Qsa, axis=-1) 
        return
    
    def sample_action_policy(self, state, policy = None):
        if policy is None: policy = self.policy
        return np.random.choice(self.actions, p= policy[state])
    
    def best_action(self, state):
        q_vals = self.Qsa[state]
        return np.random.choice(np.where(q_vals == q_vals.max())[0])

    
    def update_policy_soft(self, state, action_greedy, policy = None, eps = None):
        if eps      is None: eps    = self.eps
        if policy   is None: policy = self.policy
        policy[state]                  = np.ones(self.NUM_ACTIONS)*eps/self.NUM_ACTIONS
        policy[state][action_greedy]   = 1 - eps + eps/self.NUM_ACTIONS     

    def update_policy_Qsa(self, state, policy = None, Qsa = None, eps  = None):
        if policy is None: policy = self.policy
        if Qsa    is None: Qsa    = self.Qsa
        if eps    is None: eps    = self.eps
        policy[state] *= 0
        policy[state] += eps/self.NUM_ACTIONS
        #policy[state]                  = np.ones(self.NUM_ACTIONS)*eps/self.NUM_ACTIONS
        action_greedy                  = Qsa[state].argmax()
        policy[state][action_greedy]   =  1 - eps + eps/self.NUM_ACTIONS   
        
    def state_remap(self,state):
        if type(state) !=int and len(state) > 1:  return tuple(map(int,state))
        else:               return (int(state),)
    
    def reset(self):
        state = self.env.reset()[0]
        return self.state_remap(state)
    
    def step(self, action):
        state, reward, done = self.env.step(action)[:3]
        state               = self.state_remap(state)
        return state, reward, done
    
    def play_N_episodes(self, policy, N=100):
        if N > 0:
            if policy.shape == self.STATE_SHAPE:
                action_fn = lambda x: self.policy_greedy[x]
            else:
                action_fn = lambda x: self.sample_action_policy(x, policy)
            reward_mean = 0
            steps_mean  = 0
            for _ in range(N):
                state = self.reset()
                s = 0
                while True:
                    action = action_fn(state)
                    state, reward, done = self.step(action)
                    s += 1
                    steps_mean += 1
                    if s >= 3000:
                        reward  = -100
                        done    = True
                    reward_mean += reward

                    if done: break
                
            self.steps_per_ep[self.env_iters] = (steps_mean/N)   
            self.test_rewards[self.env_iters] = (reward_mean/N)


def plot(v_ace, v_ace_no, pol_ace, pol_ace_no, extent, lables1 = None, lables2 = None):
    xmin,xmax,ymin,ymax = extent
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={'projection': '3d'})
    sx,sy = slice(xmin,xmax),slice(ymin,ymax)
    X, Y = np.meshgrid(np.arange(v_ace.shape[0]), np.arange(v_ace.shape[1]), indexing='ij')

    if lables1 is None: lables1= ['V(s) Usable Ace','V(s) No Usable Ace']
    for i,Z,t in zip([0,1],[v_ace,v_ace_no],lables1):
        ax[0,i].set_title(t)
        ax[0,i].plot_surface(X[sx,sy], Y[sx,sy], Z[sx,sy], cmap=cm.coolwarm, linewidth=5, antialiased=True)
        ax[0,i].view_init(ax[0,0].elev, -145)
        ax[0,i].set_xlabel('Dealer\'s Showing Card')
        ax[0,i].set_ylabel('Player\'s Current Sum')
        ax[0,i].set_zlabel('State Value')
        ax[0,i].set_box_aspect((1,1,0.2))
        ax[0,i].grid(False)
        ax[0,i].set_zticks([])

    if lables2 is None: lables2= ['$\pi$(s) Ace','$\pi$(s) no Ace']   
    for i, Z, t in zip([0,1],[pol_ace, pol_ace_no],lables2):
        ax[1, i].remove()
        ax[1, i] = fig.add_subplot(2, 2, i + 3)
        ax[1,i].set_title(t)
        ax[1,i].matshow(Z[sx,sy], cmap=cm.coolwarm, extent= [ymin-0.5,ymax+0.5,xmin-0.5,xmax+0.5])#
        ax[1,i].invert_yaxis()
        ax[1,i].set_aspect(1)

    
def plot_cliff_walking(policy):
    # ChatGPT + modified
    fig, axes = plt.subplots(4, 12, figsize=(12,4), sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    axes = axes.flatten()
    for i,(ax,arrows) in enumerate(zip(axes,policy)):
        arrows2 = arrows.copy()
        arrows2[arrows<0.5] = 0
        #if all(arrows): continue
        if arrows2[3]:  # Left arrow
            ax.arrow(0.5, 0.5, -0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
        if arrows2[2]:  # Bottom arrow
            ax.arrow(0.5, 0.5, 0, -0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
        if arrows2[1]:  # Right arrow
            ax.arrow(0.5, 0.5, 0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
        if arrows2[0]:  # Top arrow
            ax.arrow(0.5, 0.5, 0, 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
