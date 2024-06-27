#from data_processing.Regression.coarse_coding_bad_approach import tiles
from data_processing.Regression.coarse_coding_tiles_ND import multi_tile
from data_processing.neural_networks.RL_Reinforced_Learning.RL_funcs import argmax_random_choice

import numpy as np, gym
#from typing import List,Tuple

class sample_space_info():
    def __init__(self, env: gym.Env, obs_space = None) -> None:
        self.obs_space      = env.observation_space if obs_space is None else obs_space
        self.obs_types      = ()
        self.obs_sizes      = ()
        # for now i am assuming that there is only one action, either discrete or cont.
        self.act_space      = env.action_space
        
            
    def process_box(self, box):
        low     = box.low
        high    = box.high
        for min_max in zip(low,high):
            self.obs_types += ('float',)
            self.obs_sizes += (min_max,)

    def process_discrete(self, discrete):
        self.obs_types += ('int',)
        self.obs_sizes += (discrete.n,)
        
    def process_tuple(self, space):
        for obs in space:
            obs_type = type(obs)
            if obs_type == gym.spaces.discrete.Discrete:
                self.process_discrete(obs)
            elif obs_type == gym.spaces.box.Box:
                self.process_box(obs)
    
    def process_states(self):
        # observation space
        if type(self.obs_space) == gym.spaces.discrete.Discrete:
            f = self.process_discrete
        elif type(self.obs_space) == gym.spaces.box.Box:
            f = self.process_box
        else: # should be tuple?
            assert type(self.obs_space) == gym.spaces.tuple.Tuple, 'unexpected state type (im dumb)'
            f = self.process_tuple
        f(self.obs_space)

        # action space
        if type(self.act_space) == gym.spaces.box.Box:
            act_type  = 'float' 
            act_size  = (self.act_space.low, self.act_space.high)
        else:
            act_type  = 'int' 
            act_size  = (self.act_space.n,)

        return self.obs_types, self.obs_sizes, act_type, act_size
    
class env_state_space_discrete():
    """ 
        Get discrete embeddings- 
        Observation (ignore actions for this example): [Discrete(A), Discrete(B),...]
        Imagine that you store OBS as an entry in array of shape (Discrete(A).n,Discrete(B).n,...) -> OBS.sample()->[1,5,...].
        We flatten this array to produce some state embedding Xs(OBS_State). But we dont really need to store this array.. just retrieve indices.
    """
    def __init__(self, state_shape) -> None:
        self.STATE_SHAPE        = state_shape
        self.NUM_STATES_TOTAL   = np.prod(state_shape)
    
    def Xs(self, state):
        return (np.ravel_multi_index(state, self.STATE_SHAPE),)

class env_state_space_continuous():
        """Get discrete embeddings: use 2D space coarse coding with overlapping tiles."""
        def __init__(self, space_types, space_lims,  tile_params) -> None:
            num_tiles, num_tilings, overlap =  tile_params
            params = []
            for lims, stype in zip(space_lims, space_types):
                if stype == 'float':
                    params.append([*lims, [num_tilings]*num_tiles, overlap])
                else:
                    params.append([0, lims-1])   # min: 0; max: n_states - 1
   
            self.tile               = multi_tile(num_tiles,space_types,params)
            
            self.NUM_STATES_TOTAL   = self.tile.max_tiling_ID

        def Xs(self, state):
            return self.tile.get_idx(state)


class base_env_class():
    """i cast actions and states to tuples. its needed when concating"""
    def __init__(self, env_name, eps, *args, **kwargs):
        self.env  = gym.make(env_name, *args, **kwargs)
        (self.space_types, self.space_lims,
         self.action_type, self.action_lims) = sample_space_info(self.env).process_states()
        self.eps = eps
        
        assert self.action_type == 'int', 'continuous actions are hard. how to max_a Q(s,a)?'

        self.NUM_ACTIONS    = self.env.action_space.n 
        self.actions        = np.arange(self.NUM_ACTIONS)
        self.policy_action  = np.zeros((self.NUM_ACTIONS,self.NUM_ACTIONS))
        self.init_policy_action_entries()
        self.env_iters      = 0

    def init_policy_action_entries(self):
        self.policy_action *= 0
        self.policy_action += self.eps/(self.NUM_ACTIONS)
        self.policy_action += np.eye(self.NUM_ACTIONS)*(1 - self.eps)

    def map_2_tuple(self,x):
        if type(x) == int:
            return (x,)
        else:
            return tuple([a for a in x])

    def reset(self):
        state = self.env.reset()[0]
        return self.map_2_tuple(state)
    
    def step(self, action):
        state, reward, done = self.env.step(action)[:3]
        return self.map_2_tuple(state), reward, done
    
   

class approach_approx_V(base_env_class):
    """interested in states only.
        Currently assuming that features are binary. X(S) is a list of active features.
        if it was one-hot encoded X(s) = [1,0,..,1] then V(s) = Wv.X(s) = W_v0 + W_vn
        but for a list X(s) = [0,n], Wv[X(s)] = [Wv_1, Wv_n], V(S) = sum(Wv[X(S)])
        List approach changes definitions a bit: weights updates as Wv[X(S)], not Wv += a*X_onehot
    """
    def __init__(self, env_name, eps, tile_params, *args, **kwargs):
        super().__init__(env_name, eps, *args, **kwargs)
        
        if all([a == 'int' for a in self.space_types] ):
            self.state_emb = env_state_space_discrete(self.space_lims)
        else:
            self.state_emb = env_state_space_continuous(self.space_types, self.space_lims, tile_params)

        self.Wv     = np.zeros(self.state_emb.NUM_STATES_TOTAL)
        self.policy = np.zeros(self.state_emb.NUM_STATES_TOTAL, dtype = int)

    def init_Vs(self, random = False, val = 0.5):
        self.Wv *= 0
        if random:  self.Wv += np.random.randn(*self.Wv.shape) 
        else:       self.Wv += val   
    
    def Vs(self,state):# not one-hot encoded. just list of active features
        return np.sum(self.Wv[self.state_emb.Xs(state)])
    
    def update_Wv(self, step_size, target, state):# change only weights active in state embedding
        self.Wv[self.state_emb.Xs(state)] += step_size*(target - self.Vs(state))

    def sample_action(self, state):  # for discrete actions
        greedy_action = self.policy[state]
        return np.random.choice(self.actions, p= self.policy_action[greedy_action])
    

class approach_approx_Q(base_env_class):
    """ For now i only know how to deal with discrete actions. 
        policy_action holds pre-baked eps-greedy transition probs for any greedy action.
        ('cause why recalculate them, if , given eps is static, they dont change.)
    """
    def __init__(self, env_name, eps, tile_params, *args, **kwargs):
        super().__init__(env_name, eps, *args, **kwargs)
        # expand state space with action space
        SA_type = self.space_types + (self.action_type,)
        SA_lims = self.space_lims + self.action_lims
        
        if all([a == 'int' for a in SA_type]):
            self.state_emb = env_state_space_discrete(SA_lims)
        else: 
            self.state_emb = env_state_space_continuous(SA_type, SA_lims, tile_params)

        self.Wq             = np.zeros(self.state_emb.NUM_STATES_TOTAL)

    def init_Qsa(self, random = False, val = 0.5):
        self.Wq *= 0
        if random:
            self.Wq += np.random.randn(*self.Wq.shape) 
        else:
            self.Wq += val

    def Qsa(self, state_action):
        return np.sum(self.Wq[self.state_emb.Xs(state_action)])  
    
    def update_Wq(self, step_size, target, state_action):
        self.Wq[self.state_emb.Xs(state_action)] += step_size*(target - self.Qsa(state_action))

    def sample_action(self, state):  # for discrete actions
        Qsa_vals = np.array([self.Qsa(state + (action,)) for action in self.actions])
        best_action = argmax_random_choice(Qsa_vals) 
        return np.random.choice(self.actions, p= self.policy_action[best_action])

class env_approx():
    def __init__(self, env_name, tile_params=None,  eps = 0.1, ALPHA = 0.05, GAMMA = 1, *args, **kwargs) -> None:
        """ For linear approximation i need state-space embeddings. 
                For discrete case:
                    default states are indexed = [0,1,...], with corresponding weights W = [w_0,w_1,...]; w_1 = W[1]
                    we embed states as one-hot vectors and retrieve via dot product W.X(1) =W.[0,1,0,...] = 0 + w_1 + 0 + ... = w_1
                For continuous case:
                    For binary encoding its the same as w_1 = W.X(1) =W.[0,1,0,...], except contents of X(1)
                So, for this part we need to get state-space embedding functions that retrieve embeddings  Xs(S) := X(S)

            grad_Qsa() and grad_Vs() are defined for linear approximation.
            Qsa() and/or Vs() can eval slices of data instead of one action/state.
            Since we use eps-greedy i store only greedy values and roll it vs remaining eps. 
            init_policy_entries() makes eps-greedy policy for all actions. ~{0:[1-eps, eps/2, eps/2],...}.
        """
        self.env                = gym.make(env_name, *args, **kwargs)
        space_types, space_lims, action_type, action_lims = sample_space_info(self.env)

        if all([a == 'int' for a in space_types] + [action_type]):
            self.xxx = env_state_space_discrete(self.env)
        else:
            self.xxx = env_state_space_continuous(space_types, space_lims, tile_params)

        self.eps                    = eps
        self.NUM_ACTIONS            = self.env.action_space.n 
        self.STATE_SHAPE_EXTENDED   = self.STATE_SHAPE + (self.NUM_ACTIONS,)

        self.Wv                     = np.zeros(self.STATE_SHAPE)
        self.Wq                     = np.zeros((self.NUM_ACTIONS,) + self.STATE_SHAPE)

        self.actions                = np.arange(self.NUM_ACTIONS)

        self.policy_action          = np.zeros((self.NUM_ACTIONS,self.NUM_ACTIONS))
        self.init_policy_entries()
        self.policy                 = np.zeros(self.STATE_SHAPE, dtype = int)

        self.env_iters              = 0

        self.GAMMA  = GAMMA
        self.ALPHA = ALPHA

    def __getattr__(self, name):
        return self.xxx.__getattribute__(name)
            
    # def Xs(self, state):
    #     return self.xxx.Xs(state)
    
    def init_policy_entries(self):
        self.policy_action *= 0
        self.policy_action += self.eps/(self.NUM_ACTIONS)
        self.policy_action += np.eye(self.NUM_ACTIONS)*(1 - self.eps)

    def init_Qsa(self, random = False, action = None, val = 0.5):
        self.Wq *= 0
        if random:
            self.Wq += np.random.randn(*self.Wq.shape) 
        else:
            if action is not None:  self.Wq[action] += val
            else:                   self.Wq         += val
    
    def init_Vs(self, random = False, val = 0.5):
        self.Wv *= 0
        if random:  self.Wv += np.random.randn(*self.Wv.shape) 
        else:       self.Wv += val             

    def init_policy(self):
        self.policy = self.Qsa(self.actions, None).squeeze(1).argmax(0) # None = :

    def Qsa(self, action, state):
        # notice we can pass slices or masks in actions/states
        return np.dot(self.Wq[action], self.Xs(state))  
    
    def Vs(self,state):
        return np.dot(self.Wv, self.Xs(state))

    def update_V(self):
        self.Wv = self.Qsa(self.actions, None).squeeze(1).max(0)
        
    def grad_Qsa(self,action,state):
        return self.Xs(state)
    
    def grad_Vs(self,state):
        return self.Xs(state)
    
    def update_Wq(self, step_size, target, state, action):
        self.Wq[action] += step_size*(target - self.Qsa(action,state))*self.grad_Qsa(self,action,state)

    def update_Wv(self, step_size, target, state):
        self.Wv += step_size*(target - self.Vs(state))*self.grad_Vs(self,state)
    
    def sample_action(self, state, use_Qsa = True):
        best_action = argmax_random_choice(self.Qsa(self.actions,state)) if use_Qsa else self.policy[state]
        return np.random.choice(self.actions, p= self.policy_action[best_action])

    # def state_remap(self,state):
    #     if type(state) !=int and len(state) > 1:  return tuple(map(int,state))
    #     else:               return (int(state),)
    
    def reset(self):
        state = self.env.reset()[0]
        #return self.state_remap(state)
        return state
    
    def step(self, action):
        state, reward, done = self.env.step(action)[:3]
        return state, reward, done
    
if __name__ == "__main__":
    if 1 == -1:
        custom_space = True
        env = gym.make('MountainCar-v0')
        if custom_space:
            obd_space1 = gym.spaces.Discrete(10)
            obs_space2 = gym.spaces.Box(low=np.array([1,2]), high=np.array([4,6]), shape=(2,), dtype=np.float32)
            obs_space = gym.spaces.Tuple((obd_space1,obs_space2)) if custom_space else None
        else: obs_space = None
        print(sample_space_info(env, obs_space).process_states())

    if 1 == -11:
        print('\nFROZEN LAKE DISCRETE 16 STATES')
        DISCRETE_ENV_V = approach_approx_V("FrozenLake-v1",eps = 0.1, tile_params=None)
        DISCRETE_ENV_V.init_Vs(random=1)
        print(f'{DISCRETE_ENV_V.Wv = }')
        print(f'{DISCRETE_ENV_V.state_emb.Xs((2,)) = }')
        print(f'{DISCRETE_ENV_V.Vs((2,)) = }')
    if 1 == -1:
        print('\nFROZEN LAKE DISCRETE 16 STATES + 4 ACTIONS')
        DISCRETE_ENV_Q = approach_approx_Q("FrozenLake-v1",eps = 0.0, tile_params=None)
        DISCRETE_ENV_Q.init_Qsa(random=1)
        print(f'{DISCRETE_ENV_Q.Wq = }')
        print(f'{DISCRETE_ENV_Q.actions = }')
        print(f'{DISCRETE_ENV_Q.state_emb.STATE_SHAPE = }')
        print(f'{DISCRETE_ENV_Q.state_emb.Xs((15,3)) = }')
        print(f'{DISCRETE_ENV_Q.Qsa((15,3)) = }')
        print(f'Q(S,a_i) = {[DISCRETE_ENV_Q.Qsa((15,a)) for a in DISCRETE_ENV_Q.actions]}')
        print(f'{DISCRETE_ENV_Q.sample_action((15,)) = }')

    tile_params = (3, 10, 0.1) #num_tiles, num_tilings, overlap
    if 1 == -1:
        print('\nMOUNTAIN CAR CONTINUOUS STATE SPACE- 2 FLOATS, 3 TILES, 10 TILINGS, 0.1 TILE OVERLAP')
        CONTI_ENV_V = approach_approx_V("MountainCar-v0",eps = 0.0, tile_params=tile_params)
        CONTI_ENV_V.init_Vs(random=1)
        state = (0.0,0.0)
        print(f'{state = }; {CONTI_ENV_V.state_emb.Xs(state) = }')
        print(f'{CONTI_ENV_V.Vs(state) = }')
    if 1 == -1:
        print('\nMOUNTAIN CAR CONTINUOUS STATE SPACE - 2 FLOATS and 1 INT ACTION SPACE, 3 TILES, 10 TILINGS, 0.1 TILE OVERLAP')
        CNT_DSC_ENV_V = approach_approx_Q("MountainCar-v0",eps = 0.0, tile_params=tile_params)
        CNT_DSC_ENV_V.init_Qsa(random=1)
        state_action = (0.0, 0.0, 0)
        state = state_action[:2]
        print(f'{CNT_DSC_ENV_V.state_emb.NUM_STATES_TOTAL = }')
        print(f'{state = }; {state_action = }; {CNT_DSC_ENV_V.state_emb.Xs(state_action) = }')
        print(f'{CNT_DSC_ENV_V.Qsa(state_action) = }')
        print(f'Q(S,a_i) = {[CNT_DSC_ENV_V.Qsa(state + (a,)) for a in CNT_DSC_ENV_V.actions]}')
        print(f'{CNT_DSC_ENV_V.sample_action(state) = }')