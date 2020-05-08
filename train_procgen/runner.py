import numpy as np
import imageio
import train_procgen.data_augs as rad
import time

from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, data_aug, eval_flag=False):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.data_aug = data_aug
        self.eval_flag = eval_flag

        # set data augmentation
        nenvs = self.obs.shape[0]
        self.augs_funcs = None
        if self.data_aug is not 'normal' and self.eval_flag is False:
            aug_to_func = {    
                'gray':rad.RandGray,
                'cutout':rad.Cutout,
                'cutout_color':rad.Cutout_Color,
                'flip':rad.Rand_Flip,
                'rotate':rad.Rand_Rotate,
                'color_jitter':rad.ColorJitterLayer,
                'crop':rad.Rand_Crop,
                }
            self.augs_funcs = aug_to_func[data_aug](batch_size=nenvs, p_gray=0.8)
            self.obs = self.augs_funcs.do_augmentation(self.obs)
            
        if self.data_aug is not 'crop' and self.eval_flag:
            self.augs_funcs = rad.Center_Crop()
            self.obs = self.augs_funcs.do_augmentation(self.obs)
            
    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for img_count in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs, rewards, self.dones, infos = self.env.step(actions)
            count = 0
            for info in infos:
                maybeepinfo = info.get('episode')

                count+=1
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            if self.data_aug is not 'normal' and self.eval_flag is False:
                self.obs[:] = self.augs_funcs.do_augmentation(obs)
            elif self.data_aug is not 'crop' and self.eval_flag:
                self.obs[:] = self.augs_funcs.do_augmentation(obs)
            else:
                self.obs[:] = obs
            
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        
        # reset random parameters    
        if self.data_aug is not 'normal' and self.eval_flag is False:
            self.augs_funcs.change_randomization_params_all()
            self.obs = self.augs_funcs.do_augmentation(obs)
        
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
    
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])