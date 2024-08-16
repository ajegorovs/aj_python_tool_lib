import gymnasium as gym, torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TRPO_env():
    """this 'wrapper' modifies env control
        automatically adds non terminal states, actions and reward into a buffer
        tracks episode rewards and calculates cumulative future rewards
        """
    def __init__(self, env: gym.Env, device = 'cpu', lam = 0.95, gamma = 0.99, **kwargs) -> None:
        #self.env = gym.make(id, **kwargs)
        self.env = env
        self.env_iters          = 0
        self.device             = device
        self.dtype              = torch.float32
        self.tensor_params      = {'dtype': self.dtype, 'device': device}
        self.tensor_params2     = {'dtype': self.dtype, 'device': 'cpu'}
        self.chunks             = []
        self.lam, self.gamma    = lam, gamma
        
    def gpu_float_tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)
    
    def init_batches(self, len_max, state_dim, action_dim):
        self.batch_states_cpu  = torch.zeros(size=(len_max+1, state_dim) , **self.tensor_params2)
        self.batch_actions_cpu  = torch.zeros(size=(len_max, action_dim), **self.tensor_params2)
        self.batch_rewards_cpu  = torch.zeros(size=(len_max, )          , **self.tensor_params2)

    def clear_batches(self):
        self.batch_states_cpu   = 0*self.batch_states_cpu   
        self.batch_actions_cpu  = 0*self.batch_actions_cpu 
        self.batch_rewards_cpu  = 0*self.batch_rewards_cpu 
        self.chunks             = []

    def reset_episode_batch(self, idx_from):
        # self.chunks is simply not called
        self.batch_states_cpu [idx_from:] *= 0 
        self.batch_actions_cpu[idx_from:] *= 0 
        self.batch_rewards_cpu[idx_from:] *= 0
                 
    def reset(self, idx):
        """ Resets env and remembers new state"""
        state = self.env.reset()[0]
        state = self.gpu_float_tensor(state)
        self.batch_states_cpu[idx] = state
        return state
    
    def step(self, action, idx):
        """ On time step remember state, action and reward received."""
        state, reward, done         = self.env.step(action)[:3]
        state   = torch.as_tensor(state, **self.tensor_params2) 
        action  = torch.as_tensor(action,**self.tensor_params2) 
        self.batch_rewards_cpu[idx]   = reward
        self.batch_actions_cpu[idx]   = action
        self.batch_states_cpu [idx+1] = state       # reset added time=0 state
        return self.gpu_float_tensor(state), done   # need NN input on gpu
    
    @torch.no_grad()
    def play_batch_terminal_only(self, policy, eps_per_batch, endless = False, max_ep_len = float('inf'), report_period = 50, tqdm_iter_params = None):
        if tqdm_iter_params is not None: tqdm_iter, tqdm_params = tqdm_iter_params
        def tqdm_report(x,y): 
            if tqdm_iter_params is not None: tqdm_iter.set_postfix(**tqdm_params, batch_size = x, epoch = y)
        self.clear_batches()
        total_successful_steps = 0
        for epoch in range(eps_per_batch):
            got_successful_termination = False
            while not got_successful_termination:
                tqdm_report(total_successful_steps,epoch)
                time = 0
                obs  = self.reset(total_successful_steps)
                while True:
                    total_steps_epoch = total_successful_steps + time
                    if time % report_period == 0: tqdm_report(total_steps_epoch,epoch)
                    action = policy.get_action(obs) 
                    obs, episode_finished = self.step(action,total_steps_epoch)
                    if not episode_finished and time + 1 >= max_ep_len:
                        if not endless:
                            self.reset_episode_batch(total_successful_steps) 
                            break
                        else: episode_finished = True
                    if episode_finished:                  
                        self.env_iters  += 1
                        self.chunks.append(time+1)  # ep length
                        got_successful_termination = True
                        total_successful_steps = total_successful_steps + time + 1 # offset +1 for next episode
                        break
                       
                    time += 1
        # ====== Trim unused entries. Terminal state should be discarded. =========       
        self.batch_states     = self.batch_states_cpu [:total_successful_steps].to(self.device)
        self.batch_actions    = self.batch_actions_cpu[:total_successful_steps].to(self.device)
        self.batch_rewards    = self.batch_rewards_cpu[:total_successful_steps].to(self.device)
        return 1

    # play_batch_REMOVED() is not used. Its not updated for reserved memory and has flawed goal logic.
    @torch.no_grad()
    def play_batch_REMOVED(self, policy, value, eps_per_batch, max_batch_size, max_ep_len = float('inf'), trunk_is_success = False,  tqdm_iter_params = None):
        """episode-per-batch-centric point of view. At early learning i want to finish set number of episodes.
            - max_batch_size is a memory buffer size. If its full stop playing episodes (leave_epoch_loop flag stops iterations).
            There are environments where, in order to learn, its necessary to terminate successfully. (we will return to this)
            - max_ep_len is manual ep truncation (replaces OG). early-terminated reward is bootstrapped with state value
            - trunk_is_success flag controls if we accept results that did not terminate correctly (no final reward)
            Sometimes first episode will fill whole buffer. If we want true termination (trunk_is_success = 0 (EDIT:correct?))
            then i will discard current data and restart epoch 1. Once i get first successful termination, i stop restarting."""
            
        if tqdm_iter_params is not None: tqdm_iter, tqdm_params = tqdm_iter_params
        max_ep_len = min(max_batch_size,max_ep_len) # if batch is smaller then max ep length
        leave_epoch_loop   = False  
        got_successful_termination   = False  # is True on first successfully terminated trajectory
        batch_L            = 0
        while not got_successful_termination:
            self.clear_batches()
            policy.clear_batches()
            for i in range(eps_per_batch):
                time = 0
                obs  = self.reset(**policy.tensor_params)
                while True:
                    if time % 50 == 0:
                        batch_L = len(self.batch_states)
                        if tqdm_iter_params is not None: tqdm_iter.set_postfix(**tqdm_params, batch_size = batch_L)
                    action = policy.get_action(obs) 
                    obs, episode_finished = self.step(action, **policy.tensor_params)
                    batch_full = len(self.batch_states) >= max_batch_size
                    episode_finished = time >= max_ep_len-1 if trunk_is_success else episode_finished
                    early_termination =  time >= max_ep_len-1 or batch_full
                    if episode_finished or early_termination:                   # if finished or big episode or batch full
                        if episode_finished: got_successful_termination = True  # cancel outer loop
                        else: self.batch_rewards[-1] += self.gamma * value(obs) # bootstrap early termination. TD1 =
                                                                                # current reward + gamma * future reward
                        if batch_full:                                          # stop collecting data (in outer loop iter)
                            leave_epoch_loop = True
                            if not got_successful_termination: break            # full batch without any finished runs. dont add iter
                                                                           
                        self.env_iters  += 1
                        self.chunks.append(time+1)
                        break
                    else:   # next state is non-terminal. add to batch
                        self.batch_states.append(obs)
                        time += 1
                    
                if leave_epoch_loop: break

        self.batch_states     = torch.stack(self.batch_states ).to(**policy.tensor_params)
        self.batch_actions    = torch.stack(self.batch_actions).to(**policy.tensor_params)
        self.batch_rewards    = torch.tensor(self.batch_rewards, **policy.tensor_params)
        return 1
    
    def rewards_2_go(self, use_gamma = False):
        self.r2g = torch.zeros_like(self.batch_rewards)
        chunks_r = torch.split(self.batch_rewards,self.chunks)
        z= 0
        for rews,c in zip(chunks_r, self.chunks):
            if use_gamma: 
                for t in range(len(rews)):
                    self.r2g[z:z+c][t] = torch.sum(rews[t:]*self.gamma**torch.arange(len(rews[t:]), device = device, dtype = torch.float32))
            else:
                self.r2g[z:z+c] = torch.flip(torch.cumsum(torch.flip(rews,(0,)),0),(0,))
            z += c
        return self.r2g
    
    @torch.no_grad()
    def advantages_GAE(self, value,**kwargs):
        chunks_Vs = torch.split(value(self.batch_states).reshape(-1), self.chunks)
        chunks_r  = torch.split(self.batch_rewards,self.chunks)
        advantages = []
        for Rw, Vs in zip(chunks_r, chunks_Vs):
            Vs = torch.cat((Vs, torch.tensor([0], **kwargs)))
            deltas = Rw + self.gamma * Vs[1:] - Vs[:-1]
            adv = torch.zeros_like(Rw, **kwargs)
            gae = 0
            for t in reversed(range(len(Rw))):
                gae = deltas[t] + self.gamma * self.lam * gae
                adv[t] = gae
            advantages.append(adv)
        return torch.cat(advantages)