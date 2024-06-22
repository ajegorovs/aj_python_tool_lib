from data_processing.Regression.coarse_coding import tiles
from data_processing.neural_networks.RL_Reinforced_Learning.RL_funcs import argmax_random_choice

import numpy as np, gym
class env_state_space_discrete():
    """Get discrete embeddings: for discrete env its trivial. Need this implementation as an alternative to continuous case."""
    def __init__(self, env) -> None:
        if type(env.observation_space) == gym.spaces.tuple.Tuple:
            self.STATE_SHAPE = tuple(map(lambda x: x.n, env.observation_space))
        else:
            self.STATE_SHAPE = (env.observation_space.n,) 
        self.NUM_STATES_TOTAL       = np.prod(self.STATE_SHAPE)
        self.M_embed                = np.eye(self.NUM_STATES_TOTAL)
    
    def Xs(self,state):
            return self.M_embed[state]

class env_state_space_continuous():
        """Get discrete embeddings: use 2D space coarse coding with overlapping tiles."""
        def __init__(self, env, tile_params_d) -> None:
            assert type(env.observation_space)      == gym.spaces.box.Box   , 'ashdas'
            assert len(env.observation_space.low)   == 2                    , 'adsdas'
            low                 = env.observation_space.low
            high                = env.observation_space.high
            minmax_x, minmax_y  = np.vstack((low,high)).T
            self.tile = tiles(**tile_params_d)
            self.tile.embedding(minmax_x,minmax_y) 
            self.STATE_SHAPE    = self.tile.M_embed[0,0].shape
            self.NUM_STATES_TOTAL       = np.prod(self.STATE_SHAPE)

        def Xs(self,state):
            return self.tile.get_embed(*state)


class env_approx():
    def __init__(self, env_name, state_space_discrete, tile_params_d=None,  eps = 0.1, ALPHA = 0.05, GAMMA = 1, *args, **kwargs) -> None:
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
        self.env                    = gym.make(env_name, *args, **kwargs)
        if state_space_discrete:
            self.xxx = env_state_space_discrete(self.env)
        else:
            self.xxx = env_state_space_continuous(self.env,tile_params_d)

        self.STATE_SHAPE            = self.xxx.STATE_SHAPE
        self.NUM_STATES_TOTAL       = self.xxx.NUM_STATES_TOTAL
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

    def state_remap(self,state):
        if type(state) !=int and len(state) > 1:  return tuple(map(int,state))
        else:               return (int(state),)
    
    def reset(self):
        state = self.env.reset()[0]
        #return self.state_remap(state)
        return state
    
    def step(self, action):
        state, reward, done = self.env.step(action)[:3]
        #state               = self.state_remap(state)
        return state, reward, done
    
# class env_approx():
#     def __init__(self, env_name, state_embeddings = None,  eps = 0.1, ALPHA = 0.05, GAMMA = 1, *args, **kwargs) -> None:
#         """     grad_Qsa() and grad_Vs() are defined for linear approximation.
#                 Qsa() and/or Vs() can eval slices of data instead of one action/state.
#                 Since we use eps-greedy i store only greedy values and roll it vs remaining eps. 
#                 init_policy_entries() makes eps-greedy policy for all actions. ~{0:[1-eps, eps/2, eps/2],...}.
#         """
#         self.env                    = gym.make(env_name, *args, **kwargs)
#         # if space is multi-dimensional -> tuples of spaces
#         if type(self.env.observation_space) == gym.spaces.tuple.Tuple:
#             self.STATE_SHAPE        = tuple(map(lambda x: x.n, self.env.observation_space))
#         else:
#             self.STATE_SHAPE        = (self.env.observation_space.n,) 

#         self.eps                    = eps
#         self.NUM_ACTIONS            = self.env.action_space.n 
#         self.NUM_STATES_TOTAL       = np.prod(self.STATE_SHAPE)
#         self.STATE_SHAPE_EXTENDED   = self.STATE_SHAPE + (self.NUM_ACTIONS,)

#         self.Wv                     = np.zeros(self.STATE_SHAPE)
#         self.Wq                     = np.zeros((self.NUM_ACTIONS,) + self.STATE_SHAPE)

#         if state_embeddings is None:
#             self.Xs                 = np.eye(self.NUM_STATES_TOTAL)
#         else:   
#             self.Xs                 = state_embeddings
#         self.actions                = np.arange(self.NUM_ACTIONS)

#         self.policy_action          = np.zeros((self.NUM_ACTIONS,self.NUM_ACTIONS))
#         self.policy                 = np.zeros(self.STATE_SHAPE, dtype = int)

#         self.env_iters              = 0

#         self.GAMMA  = GAMMA
#         self.ALPHA = ALPHA
            

#     def init_policy_entries(self):
#         self.policy_action *= 0
#         self.policy_action += self.eps/(self.NUM_ACTIONS)
#         self.policy_action += np.eye(self.NUM_ACTIONS)*(1 - self.eps)

#     def init_Qsa(self, random = False, action = None, val = 0.5):
#         self.Wq *= 0
#         if random:
#             self.Wq += np.random.randn(*self.Wq.shape) 
#         else:
#             if action is not None:  self.Wq[action] += val
#             else:                   self.Wq         += val
    
#     def init_Vs(self, random = False, val = 0.5):
#         self.Wv *= 0
#         if random:  self.Wv += np.random.randn(*self.Wv.shape) 
#         else:       self.Wv += val             

#     def init_policy(self):
#         self.policy = self.Qsa(self.actions, None).squeeze(1).argmax(0) # None = :

#     def Qsa(self, action, state):
#         # notice we can pass slices or masks in actions/states
#         return np.dot(self.Wq[action], self.Xs[state])  
    
#     def Vs(self,state):
#         return np.dot(self.Wv, self.Xs[state])

#     def update_V(self):
#         self.Wv = self.Qsa(self.actions, None).squeeze(1).max(0)
        
#     def grad_Qsa(self,action,state):
#         return self.Xs[state]
    
#     def grad_Vs(self,state):
#         return self.Xs[state]
    
#     def update_Wq(self, step_size, target, state, action):
#         self.Wq[action] += step_size*(target - self.Qsa(action,state))*self.grad_Qsa(self,action,state)

#     def update_Wv(self, step_size, target, state):
#         self.Wv += step_size*(target - self.Vs(state))*self.grad_Vs(self,state)
    
#     def sample_action(self, state, use_Qsa = True):
#         best_action = argmax_random_choice(self.Qsa(self.actions,state)) if use_Qsa else self.policy[state]
#         return np.random.choice(self.actions, p= self.policy_action[best_action])

#     def state_remap(self,state):
#         if type(state) !=int and len(state) > 1:  return tuple(map(int,state))
#         else:               return (int(state),)
    
#     def reset(self):
#         state = self.env.reset()[0]
#         return self.state_remap(state)
    
#     def step(self, action):
#         state, reward, done = self.env.step(action)[:3]
#         state               = self.state_remap(state)
#         return state, reward, done
    

class Qsa_APPROX():
    def __init__(self, env, tile_params_d) -> None:
        assert type(env.observation_space)      == gym.spaces.box.Box , 'ashdas'
        assert len(env.observation_space.low)   == 2, 'adsdas'
        self.env            = env
        low                 = env.observation_space.low
        high                = env.observation_space.high
        minmax_x, minmax_y  = np.vstack((low,high)).T
        self.NUM_ACTIONS    = env.action_space.n
        self.actions = np.arange(self.NUM_ACTIONS)
        self.init_tile(tile_params_d, minmax_x, minmax_y)
        self.w = np.zeros(shape=(self.actions.size, self.num_f))

    def init_tile(self, tile_params_d, dom_x, dom_y):
        self.tile = tiles(**tile_params_d)
        self.tile.embedding(dom_x,dom_y)      
        self.num_f = self.tile.get_embed(0,0).size  # might break if out of domain

    def xSt(self,state):
        return self.tile.get_embed(*state)
    
    def Qsa(self, action, state):
        return np.dot(self.w[action], self.xSt(state))
    
    def update_w(self, alpha, target, state, action):
        self.w[action] += alpha*(target - self.Qsa(action,state))*self.xSt(state)

