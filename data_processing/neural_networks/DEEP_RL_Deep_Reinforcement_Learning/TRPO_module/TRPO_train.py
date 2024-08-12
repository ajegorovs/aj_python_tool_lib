from tqdm import tqdm
import torch, os
from data_processing.neural_networks.DEEP_RL_Deep_Reinforcement_Learning.TRPO_module.TRPO_CG import conjugate_gradient_hess
from data_processing.neural_networks.DEEP_RL_Deep_Reinforcement_Learning.TRPO_module.TRPO_brain import TRPO_Proto_Policy, value
from data_processing.neural_networks.DEEP_RL_Deep_Reinforcement_Learning.TRPO_module.TRPO_brain import Policy_Discrete, Policy_continuous
from data_processing.neural_networks.DEEP_RL_Deep_Reinforcement_Learning.TRPO_module.TRPO_env import TRPO_env
from gymnasium.spaces import Box, Discrete  # cannot reuse gymnasium as gym ;(. refers to old gym

EPS = 1e-8

def update_model_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        #p.data.copy_(flat_params[offset:offset+numel].view_as(p))
        p.data += flat_params[offset:offset+numel].view_as(p)
        offset += numel

class TRPO_train_wrap():
    def __init__(self, env: TRPO_env, actor_hidden = [32], critic_hidden = [32], lr = 0.01) -> None:
        self.env        = env
        self.num_obs    = env.env.observation_space.shape[0]
        self.critic     = value(self.num_obs, 1 , critic_hidden, lr = lr)
        if isinstance(env.env.action_space, Box):
            self.dim_acts   = self.num_acts = env.env.action_space.shape[0]
            self.actor      = Policy_continuous(self.num_obs, self.num_acts, actor_hidden)
        elif isinstance(env.env.action_space, Discrete):
            self.num_acts        = env.env.action_space.n
            self.dim_acts   = 1
            self.actor      = Policy_Discrete(self.num_obs, self.num_acts, actor_hidden)
        else:
            raise Exception("Only continuous or only discrete actions accepted!")
        
        self.progress     = []
        self.progress_std = []
        self.step_lengths = []
        


    def save_weights(self, case_name, base_path = None):
        if base_path is None:
            base_path = os.path.join('data_processing', 'neural_networks', 'DEEP_RL_Deep_Reinforcement_Learning','TRPO_Report')
        torch.save(self.actor.mlp.state_dict(), os.path.join(base_path,case_name+'.pt'))

    def init_train_params(self, eps_per_batch   = 3   , endless         = False,max_ep_len      = 15000,
                                delta           = 1e-2, use_gamma       = True, backtrack_coeff = 0.8,
                                backtrack_iters = 10  , damping         = 0.1 , use_FIM         = False,
                                use_CG          = True, CG_Iters        = 20  , inverse_regularization  = 0.001):
        self.eps_per_batch  = eps_per_batch
        self.endless        = endless
        self.max_ep_len     = max_ep_len
        self.delta          = delta
        self.use_gamma      = use_gamma
        self.damping        = damping
        self.use_FIM        = use_FIM
        self.use_CG         = use_CG
        self.CG_Iters       = CG_Iters
        self.backtrack_coeff        = backtrack_coeff
        self.backtrack_iters        = backtrack_iters
        self.inverse_regularization = inverse_regularization
        

    def train(self, num_iters = 30):
        #print("Training on env", self.env.env.spec.id)
        
        step_len        = lambda k: self.backtrack_coeff**k*(1-self.backtrack_coeff)
        tq_iter         = tqdm(range(num_iters))
        tq_prms         = { '1.avg_cum_reward_mean'  :0, '2.avg_cum_reward_std'   :0, 
                            '3. step_len'            :0, 'evals'                  :self.env.env_iters}
        for _ in tq_iter: 
            self.env.init_batches(self.eps_per_batch*self.max_ep_len, self.num_obs, self.dim_acts)
            # ========= Gather successful episodes =============
            self.env.play_batch_terminal_only(self.actor, self.eps_per_batch,  self.endless , self.max_ep_len, 
                                              report_period = 50, tqdm_iter_params = (tq_iter,tq_prms))
            # ========= Calculate cumulative rewards =============
            advantages = self.env.advantages_GAE(self.critic, **self.actor.tensor_params)
            rewards_to_go = self.env.rewards_2_go(self.use_gamma)
            # ========= Fit Value function =============
            self.critic.train(self.env.batch_states, rewards_to_go, n_iters=1000) # fit state value function
            # ========= Prepare 'new' and old policy =============
            self.actor.batch_calc_logPs(self.env.batch_states, self.env.batch_actions)
            self.actor.prep_old_policy()
            # ====== First iter grad reduces to NPG grad ==========
            self.actor.calc_NPG_logP_grads()
            self.actor.grad = self.actor.calc_NPG_grad(advantages)
            # ========== Fisher Information Matrix approach ===============
            if self.use_FIM:
                FIM  = self.actor.calf_FIM(self.inverse_regularization)
                if self.use_CG: # only approximate inversion
                    Hv = lambda v: FIM @ v
                    x = conjugate_gradient_hess(Hv, self.actor.grad, max_iters=self.CG_Iters)
                else:
                    FIM_inv = torch.linalg.inv(FIM)
                    x = FIM_inv @ self.actor.grad
                    del FIM_inv                 # just experimenting
                    torch.cuda.empty_cache()
            # ================= Hessian Matrix approach ==================
            else:
                # Dont calculate H inverse, only final direction
                if self.use_CG:  
                    self.actor.D_KL = self.actor.calc_KL_div()
                    
                    def Fvp(v):
                        flat_grad_kl        = self.actor.calc_flat_grad(self.actor.D_KL, create_graph=True)
                        flat_grad_grad_kl   = self.actor.calc_flat_grad(flat_grad_kl @ v)
                        return flat_grad_grad_kl + v * self.damping
                    
                    x = conjugate_gradient_hess(Fvp, self.actor.grad, max_iters=self.CG_Iters)

                # Calculate full Hessian and invert it
                else:       
                    H = self.actor.calc_Hessian(self.inverse_regularization)
                    H_inv = torch.linalg.inv(H)
                    x = H_inv @ self.actor.grad 
                    del H, H_inv
                    torch.cuda.empty_cache()

            # ====== Surrogate reward for backtracking reference ==========
            self.actor.U_rew     = advantages.flatten().mean(dim = 0)    # when old=new ratio is 1.
            self.actor.U_rew_old = self.actor.U_rew.clone().detach()  # dont need grad if we calc NPG grad
            # ====== Update policy to NPG solution ==========
            alpha = torch.sqrt(2*self.delta/(torch.dot(self.actor.grad,x)+EPS))   # NPG step length
            update_model_params(self.actor.mlp, alpha * x)                   # set NN params using NPG step
            # ======================= Backtracking Line Search ====================
            # WHAT: Eval. surrogate reward and KL divergence at NPG step params. Backtrack to old params.
            with torch.no_grad():
                backwards_fraction_traveled = 0
                for j in range(self.backtrack_iters):
                    # reevaluate probs using new policy 
                    self.actor.batch_calc_logPs(self.env.batch_states, self.env.batch_actions)
                    # update Surrogate reward and KL divergence for new policy
                    self.actor.U_rew = self.actor.calc_surrogate_reward(advantages)
                    d_reward            = self.actor.U_rew - self.actor.U_rew_old  
                    dkl                 = self.actor.calc_KL_div()

                    if dkl <= self.delta and d_reward >= 0: 
                        self.step_lengths.append(1 - backwards_fraction_traveled)
                        break # accept update
                    elif    j==self.backtrack_iters-1:   # max iters reached. reroll weights back.
                            print(f'Could not converge in {self.backtrack_iters} iterations! Restarting experiment.', end= '\r')
                            update_model_params(self.actor.mlp, -(1-backwards_fraction_traveled)*alpha * x ) # finish path to \theta_0
                            self.step_lengths.append(1 - backwards_fraction_traveled)
                            break
                    else:
                        step_size = step_len(j)
                        update_model_params(self.actor.mlp, -step_size *alpha * x )
                        backwards_fraction_traveled += step_size
                    
            # =============== Calc stats ==================
            if 1 == 1:
                chunks_r    = torch.split(self.env.batch_rewards,self.env.chunks)       
                ep_rew_sum  = torch.tensor([rew_ep.sum() for rew_ep in chunks_r])
                avg_cum_reward_std, avg_cum_reward_mean = torch.std_mean(ep_rew_sum)
                avg_cum_reward_std, avg_cum_reward_mean = avg_cum_reward_std.item(), avg_cum_reward_mean.item()
                step = (1-backwards_fraction_traveled)
                tq_prms = {'1.avg_cum_reward_mean'  :avg_cum_reward_mean, 
                        '2.avg_cum_reward_std'   :avg_cum_reward_std, 
                        '3. step_len'            :step, 'evals':self.env.env_iters}
                tq_iter.set_postfix(**tq_prms)
                self.progress.append(avg_cum_reward_mean)
                self.progress_std.append(avg_cum_reward_std)
                self.actor.grad = None

if __name__ == "__main__":
    if 1 == -1:
        import gymnasium as gym
        env_params  = {'id':"Acrobot-v1"} # (bad, hard to terminate) https://www.gymlibrary.dev/environments/classic_control/acrobot/
        env         = gym.make(**env_params)
    else:
        from gym_jiminy.envs import AcrobotJiminyEnv
        env = AcrobotJiminyEnv(False)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_CP      = TRPO_env(env, device=device)
    trainer     = TRPO_train_wrap(env_CP)
    trainer.init_train_params()
    trainer.train(num_iters=2)

