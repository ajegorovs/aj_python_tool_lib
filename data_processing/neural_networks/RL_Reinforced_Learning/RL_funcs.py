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
        action_greedy                  = np.random.choice(np.where(Qsa[state] == Qsa[state].max())[0])
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

    
def plot_cliff_walking(policy,figsize=(12,4), aw = 0.2,hw = 0.3, hl = 0.25):
    # ChatGPT + modified
    fig, axes = plt.subplots(4, 12, figsize=figsize, sharex=True,sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    axes = axes.flatten()
    for i,(ax,arrows) in enumerate(zip(axes,policy)):
        if sum(arrows) == 0: continue
        arrows = arrows.copy()
        arrows /= arrows.max()
        p = lambda i :arrows[i]
        ax.arrow(0.5, 0.5, 0, p(0)*aw , head_width=p(0)*hw, head_length=p(0)*hl, fc='k', ec='k')
        ax.arrow(0.5, 0.5, p(1)*aw, 0 , head_width=p(1)*hw, head_length=p(1)*hl, fc='k', ec='k')
        ax.arrow(0.5, 0.5, 0, -p(2)*aw, head_width=p(2)*hw, head_length=p(2)*hl, fc='k', ec='k')
        ax.arrow(0.5, 0.5, -p(3)*aw, 0, head_width=p(3)*hw, head_length=p(3)*hl, fc='k', ec='k')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def rew_ep_plot(env, fltr = 0):
    plot_data = [env.test_rewards,env.steps_per_ep]
    plot_y_lab= ['Mean cumulative reward','Mean number of steps per episode']
    fig, axs = plt.subplots(1,2, figsize = (10,3))
    for i, (ax, data, ylab) in enumerate(zip(axs,plot_data,plot_y_lab)):
        ax.plot(*np.array([(i,r) for i,r in data.items() if i >fltr]).T)
        ax.set_xlabel('# episodes')
        ax.set_ylabel(ylab)


def argmax_random_choice(array):
    results = np.zeros(shape=array.shape[:-1], dtype=int)
    for index in np.ndindex(results.shape):
        vals = array[index]
        results[index] = np.random.choice(np.where(vals == vals.max())[0])
    return results


class base_env():
    def __init__(self, env_name, *args, eps = 0.1,  **kwargs) -> None:
        self.env                    = gym.make(env_name, *args, **kwargs)
        # if space is multi-dimensional -> tuples of spaces
        if type(self.env.observation_space) == gym.spaces.tuple.Tuple:
            self.STATE_SHAPE        = tuple(map(lambda x: x.n, self.env.observation_space))
        else:
            self.STATE_SHAPE        = (self.env.observation_space.n,) 

        self.eps                    = eps
        self.NUM_ACTIONS            = self.env.action_space.n 
        self.STATE_SHAPE_EXTENDED   = self.STATE_SHAPE + (self.NUM_ACTIONS,)
        self.actions                = np.arange(self.NUM_ACTIONS)
        self.policy, self.Qsa       = np.zeros(shape= (2,) + self.STATE_SHAPE_EXTENDED)
        self.Vs                     = np.zeros(shape= self.STATE_SHAPE)
        self.NUM_STATES_TOTAL       = np.prod(self.STATE_SHAPE)
        self.test_rewards           = defaultdict(float)
        self.steps_per_ep           = defaultdict(float)
        self.env_iters              = 0
            
    def Qsa_init(self, Qsa = None, random = False, action = None, val = 0.5):
        Qsa  = self.Qsa if Qsa is None else Qsa
        Qsa *= 0
        if random:
            Qsa += np.random.randn(*Qsa.shape) 
        else:
            if action is not None:  Qsa[...,action] += val
            else:                   Qsa             += val

        return

    def policy_update_whole(self, use_Qsa = True, action = None, eps = None):
        eps = self.eps if eps is None else eps
        if use_Qsa:             # extract greedy action from Q(s,a)
            greedy =  argmax_random_choice(self.Qsa).flatten()
        elif action is None:    # random action
            greedy = np.random.randint(0, self.NUM_ACTIONS, size = (self.NUM_STATES_TOTAL,))
        else:                   # specific action
            assert type(action) == int, 'Action should be an integer index!'
            greedy  = np.array([action]*self.NUM_STATES_TOTAL)
        self.policy *= 0        # reset
        self.policy += eps/(self.NUM_ACTIONS)
        self.policy[np.arange(self.NUM_STATES_TOTAL),greedy] = (1 - eps + eps/(self.NUM_ACTIONS))

        return

    def best_action(self, state):   # with random tie break
        return argmax_random_choice(self.Qsa[state])#.flatten()  #array(3) - > array([3])
    
    def sample_action(self, state, use_Qsa = True, eps = None):
        eps = self.eps if eps is None else eps
        if use_Qsa:
            action = self.best_action(state) # with random tie break
            policy = np.ones_like(self.actions)*eps/(self.NUM_ACTIONS)
            policy[action] = (1 - eps + eps/(self.NUM_ACTIONS))
        else:
            policy = self.policy[state]

        return np.random.choice(self.actions, p= policy)
    
    def update_policy_via_Qsa(self,state, eps = None):
        eps = self.eps if eps is None else eps
        action = self.best_action(state) # with random tie break
        policy = np.ones_like(self.actions)*eps/(self.NUM_ACTIONS)
        policy[action] = (1 - eps + eps/(self.NUM_ACTIONS))
        self.policy[state] = policy

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
    
    def play_N_episodes(self, use_Qsa = False, N=100):
        # pre-compute and use policy. 'cause its faster.
        if not use_Qsa: self.policy_update_whole(use_Qsa = True)  # <<< gen pol from Q(s,a)
        if N > 0:
            reward_mean = 0
            steps_mean  = 0
            for _ in range(N):
                state = self.reset()
                s = 0
                while True:
                    action = self.sample_action(state, use_Qsa = use_Qsa)
                    state, reward, done = self.step(action)
                    steps_mean += 1
                    reward_mean += reward
                    s += 1
                    if done or s > 6000: break
                
            self.steps_per_ep[self.env_iters] = (steps_mean/N)   
            self.test_rewards[self.env_iters] = (reward_mean/N)


def anim(store_Qs):
    import matplotlib.animation as animation

    sorted_time_steps = list(store_Qs.keys())
    fig, ax = plt.subplots(figsize=(8,3))
    cax = ax.matshow(store_Qs[sorted_time_steps[0]], cmap='viridis')
    fig.colorbar(cax, location='bottom')
    ax.set_xticks([])
    ax.set_yticks([])

    def init():
        cax.set_array(store_Qs[sorted_time_steps[0]])
        return [cax]

    def update(frame):
        matrix = store_Qs[sorted_time_steps[frame]]
        cax.set_array(matrix)
        fig.suptitle(f'episode: {sorted_time_steps[frame]}')
        return [cax]

    ani = animation.FuncAnimation(fig, update, frames=len(sorted_time_steps), init_func=init, interval=500, blit=True)
    return ani