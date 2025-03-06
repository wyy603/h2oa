# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
# import ipdb

# from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticTransformer
# from rsl_rl.env import VecEnv
from legged_gym.envs.h1.h1 import H1
import IPython; e = IPython.embed

import wandb

class OnPolicyRunner:

    def __init__(self,
                 env: H1,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        if self.env.cfg.env.policy_name == "ActorCritic":
            actor_critic_class = eval("ActorCritic") # ActorCritic
            actor_critic: ActorCritic = actor_critic_class( 
                                                        self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg
                                                        ).to(self.device)
        else:
            actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
            actor_critic: ActorCriticTransformer = actor_critic_class( 
                                                        self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        self.env.obs_context_len,
                                                        **self.policy_cfg
                                                        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg = PPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        if self.env.cfg.env.policy_name == "ActorCritic":
            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])
        else:
            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.obs_context_len, self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        # ipdb.set_trace()
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        ep_metrics = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        donebuffer = deque(maxlen=100)
        termination_contact_buffer = deque(maxlen=100)
        r_threshold_buffer = deque(maxlen=100)
        p_threshold_buffer = deque(maxlen=100)
        z_threshold_buffer = deque(maxlen=100)
        fallen_buffer = deque(maxlen=100)
        time_out_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'episode_metrics' in infos:
                            ep_metrics.append(infos['episode_metrics'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        donebuffer.append(len(new_ids) / self.env.num_envs)
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        termination_contact_buffer.append(len((infos["termination"]['termination_contact_buf'] > 0).nonzero(as_tuple=False))/ self.env.num_envs)
                        r_threshold_buffer.append(len((infos["termination"]['r_threshold_buff'] > 0).nonzero(as_tuple=False))/ self.env.num_envs)
                        p_threshold_buffer.append(len((infos["termination"]['p_threshold_buff'] > 0).nonzero(as_tuple=False))/ self.env.num_envs)
                        z_threshold_buffer.append(len((infos["termination"]['z_threshold_buff'] > 0).nonzero(as_tuple=False))/ self.env.num_envs)
                        fallen_buffer.append(len((infos["termination"]['has_fallen'] > 0).nonzero(as_tuple=False))/ self.env.num_envs)
                        time_out_buffer.append(len((infos["termination"]['time_out_buf'] > 0).nonzero(as_tuple=False))/ self.env.num_envs)
                        if 'scaling_factor' in infos:
                            scaling_factor = infos['scaling_factor']
                        episode_length_current = infos['episode_length_current']

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0 and it > 0:
                    save_path = os.path.join(self.log_dir, 'model_{}.pt'.format(it))
                    self.save(save_path)
                    # clean last ckpt, remain if it in model_it.pt % 10000 == 0
                    from glob import glob
                    checkpoint_files = glob(os.path.join(self.log_dir, 'model_*.pt'))
                    if checkpoint_files:
                        for ckpt in checkpoint_files:
                            last_it =int(ckpt.split('_')[-1].split('.')[0])
                            if last_it != it and last_it % 10000 != 0:
                                os.remove(ckpt)

            ep_infos.clear()
            ep_metrics.clear()
        
        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb.log({'Episode/' + key: value}, step=locs['it'])
                wandb_dict['Episode/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if locs['ep_metrics']:
            for key in locs['ep_metrics'][0]:
                info = []
                for ep_metric in locs['ep_metrics']:
                    info.append(ep_metric[key])
                value = np.mean(info)
                # wandb.log({'Episode/' + key: value}, step=locs['it'])
                wandb_dict['Metric/' + key] = value
                ep_string += f"""{f'Mean episode metric {key}:':>{pad}} {value:.4f}\n"""
        
        std = self.alg.actor_critic.std.cpu().detach().numpy()
        mean_std = std.mean()
        entropy = self.alg.actor_critic.entropy.detach().mean().item()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        std_numpy = self.alg.actor_critic.std.cpu().detach().numpy()

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/entropy'] = entropy
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        wandb_dict['Policy/noise_std_dist'] = wandb.Histogram(std_numpy)
        wandb_dict['Std/mean_std'] = mean_std
        if 'scaling_factor' in locs:
            wandb_dict['Train/scaling_factor'] = locs['scaling_factor']
        wandb_dict['Train/episode_length_current'] = locs['episode_length_current']
        # log all dim of the std
        # for i, std in enumerate(self.alg.actor_critic.std):
        #     wandb_dict[f'Std/std_dim_{i}'] = std
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            # wandb_dict['Train/mean_arm_reward'] = statistics.mean(locs['armrewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['termination/dones'] = statistics.mean(locs['donebuffer'])
            wandb_dict['termination/termination_contact_buffer'] = statistics.mean(locs['termination_contact_buffer'])
            wandb_dict['termination/r_threshold_buffer'] = statistics.mean(locs['r_threshold_buffer'])
            wandb_dict['termination/p_threshold_buffer'] = statistics.mean(locs['p_threshold_buffer'])
            wandb_dict['termination/z_threshold_buffer'] = statistics.mean(locs['z_threshold_buffer'])
            wandb_dict['termination/fallen_buffer'] = statistics.mean(locs['fallen_buffer'])
            wandb_dict['termination/time_out_buffer'] = statistics.mean(locs['time_out_buffer'])
            # wandb.log({'Train/mean_reward/time': statistics.mean(locs['rewbuffer'])}, step=self.tot_time)
            # wandb.log({'Train/mean_episode_length/time': statistics.mean(locs['lenbuffer'])}, step=self.tot_time)
        
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
        print(self.log_dir)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
