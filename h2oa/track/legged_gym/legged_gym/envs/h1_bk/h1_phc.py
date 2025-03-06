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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os, sys
from copy import deepcopy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# import ipdb
from torch import Tensor
from typing import Tuple, Dict
import time

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, euler_from_quat
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.human import load_target_jt, load_target_body, load_target_root, load_target_pkl, load_target_pkl_concat
from legged_gym.utils.diff_quat import flip_quat_by_w, quat_to_vec6d, quat_multiply, quat_inv, broadcast_quat_apply, broadcast_quat_multiply, vec6d_to_quat
from legged_gym.utils import diff_rot
from .h1_phc_config import H1RoughCfg

def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x) + 1 if np.random.rand() < (x - int(x)) else int(x)

class H1():
    def __init__(self, cfg: H1RoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = hasattr(self.cfg,'play') and self.cfg.play
        if hasattr(self.cfg,'test') and self.cfg.test:
            self.test_motion_id = 0.
        self.init_done = False
        self._parse_cfg(self.cfg)
        # breakpoint()
        self._super_init(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        
        # human retargeted poses
        if hasattr(self.cfg.human, 'multi_motion') and self.cfg.human.multi_motion:
            self._init_target_pkls()
        else:
            self._init_target_jt()

        self.init_done = True
    
    def _super_init(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = 0
        ############### state ################
        if self.cfg.env.obs.dof_pos:
            self.num_obs += self.cfg.env.num_dofs
        if self.cfg.env.obs.dof_vel:
            self.num_obs += self.cfg.env.num_dofs

        if self.cfg.env.obs.body_pos:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.body_ori:
            self.num_obs += 6*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.body_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.body_ang_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)

        if self.cfg.env.obs.root_pos:
            self.num_obs += 3
        if self.cfg.env.obs.root_ori:
            self.num_obs += 6
        if self.cfg.env.obs.root_vel:
            self.num_obs += 3
        if self.cfg.env.obs.root_ang_vel:
            self.num_obs += 3
        if self.cfg.env.obs.root_high:
            self.num_obs += 1

        if self.cfg.env.obs.last_action:
            self.num_obs += self.cfg.env.num_dofs

        if self.cfg.env.obs.base_orn_rp:
            self.num_obs += 2
        if self.cfg.env.obs.base_lin_vel:
            self.num_obs += 3
        if self.cfg.env.obs.base_ang_vel:
            self.num_obs += 3
        if self.cfg.env.obs.projected_gravity:
            self.num_obs += 3
        if self.cfg.env.obs.commands:
            self.num_obs += 3

        ########### target ###########

        if self.cfg.env.obs.target_dof_pos:
            self.num_obs += self.cfg.env.num_dofs
        if self.cfg.env.obs.target_dof_vel:
            self.num_obs += self.cfg.env.num_dofs

        if self.cfg.env.obs.target_body_pos:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.target_body_ori:
            self.num_obs += 6*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.target_body_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.target_body_ang_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        
        if self.cfg.env.obs.target_global_pos:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.target_global_ori:
            self.num_obs += 6*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.target_global_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.target_global_ang_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        
        if self.cfg.env.obs.diff_local_pos:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.diff_local_ori:
            self.num_obs += 6*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.diff_local_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.diff_local_ang_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        
        if self.cfg.env.obs.diff_global_pos:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.diff_global_ori:
            self.num_obs += 6*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.diff_global_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        if self.cfg.env.obs.diff_global_ang_vel:
            self.num_obs += 3*(self.cfg.env.num_dofs+1)
        
        if self.cfg.env.obs.target_root_pos:
            self.num_obs += 3
        if self.cfg.env.obs.target_root_ori:
            self.num_obs += 6
        if self.cfg.env.obs.target_root_vel:
            self.num_obs += 3
        if self.cfg.env.obs.target_root_ang_vel:
            self.num_obs += 3
    
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.obs_context_len = cfg.env.obs_context_len

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs
        self.obs_history_buf = torch.zeros(self.num_envs, self.obs_context_len, self.num_obs, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        
        self.scaling_factor = 0.1

    def get_observations(self):  
        if self.cfg.env.policy_name == "ActorCritic":
            return self.obs_buf
        else:
            return self.obs_history_buf
    
    def get_privileged_observations(self):
        return None


    def _init_target_jt(self):
        self.target_jt_pos_seq, self.target_jt_vel_seq, self.target_jt_seq_len = load_target_jt(self.device, "jt_" + self.cfg.human.filename)
        # self.target_body_pos_seq, self.target_body_ori_seq, self.target_body_vel_seq, self.target_body_ang_vel_seq = load_target_body(self.device, "body_" + self.cfg.human.filename)
        if self.cfg.human.load_global:
            self.target_global_pos_seq, self.target_global_ori_seq, self.target_global_vel_seq, self.target_global_ang_vel_seq = load_target_body(self.device, "global_" + self.cfg.human.filename)
        # self.target_root_pos_seq, self.target_root_ori_seq, self.target_root_vel_seq, self.target_root_ang_vel_seq = load_target_root(self.device, "root_" + self.cfg.human.filename)
        self.num_target_jt_seq, self.max_target_jt_seq_len, self.dim_target_jt = self.target_jt_pos_seq.shape
        print(f"Loaded target joint trajectories of shape {self.target_jt_pos_seq.shape}")
        # assert(self.dim_target_jt == self.num_dofs)
        self.target_jt_i = torch.randint(0, self.num_target_jt_seq, (self.num_envs,), device=self.device)
        self.target_jt_j = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_jt_dt = 1 / self.cfg.human.freq
        self.target_jt_update_steps = self.target_jt_dt / self.dt # not necessary integer
        assert(self.dt <= self.target_jt_dt)
        self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)

        self.delayed_obs_target_jt_pos = None
        self.delayed_obs_target_jt_vel = None

        self.delayed_obs_target_body_pos = None
        self.delayed_obs_target_body_ori = None
        self.delayed_obs_target_body_vel = None
        self.delayed_obs_target_body_ang_vel = None

        self.delayed_obs_target_root_pos = None
        self.delayed_obs_target_root_ori = None
        self.delayed_obs_target_root_vel = None
        self.delayed_obs_target_root_ang_vel = None

        self.delayed_obs_target_jt_steps = self.cfg.human.delay / self.target_jt_dt
        self.delayed_obs_target_jt_steps_int = sample_int_from_float(self.delayed_obs_target_jt_steps)
        self.update_target_jt(torch.tensor([], dtype=torch.long, device=self.device))

    def _init_target_pkls(self):
        self.target_jt_seq, self.target_global_seq, self.target_seq_len, self.motion_start_id = load_target_pkl_concat(self.device, self.cfg.human.filename)
        self.num_target_seq = self.target_seq_len.shape[0]
        print(f"Loaded target joint trajectories of shape {self.target_seq_len.shape}")
        # assert(self.dim_target_jt == self.num_dofs)
        if hasattr(self.cfg,'test') and self.cfg.test:
            self.target_jt_i = torch.arange(self.num_envs, device=self.device) % self.num_target_seq
        else:
            self.target_jt_i = torch.randint(0, self.num_target_seq, (self.num_envs,), device=self.device)
        self.target_jt_j = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_jt_dt = 1 / self.cfg.human.freq
        self.target_jt_update_steps = self.target_jt_dt / self.dt # not necessary integer
        assert(self.dt <= self.target_jt_dt)
        self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)

        self.delayed_obs_target_jt_pos = None
        self.delayed_obs_target_jt_vel = None

        self.delayed_obs_target_body_pos = None
        self.delayed_obs_target_body_ori = None
        self.delayed_obs_target_body_vel = None
        self.delayed_obs_target_body_ang_vel = None

        self.delayed_obs_target_root_pos = None
        self.delayed_obs_target_root_ori = None
        self.delayed_obs_target_root_vel = None
        self.delayed_obs_target_root_ang_vel = None

        self.delayed_obs_target_jt_steps = self.cfg.human.delay / self.target_jt_dt
        self.delayed_obs_target_jt_steps_int = sample_int_from_float(self.delayed_obs_target_jt_steps)
        self.update_target_pkls(torch.tensor([], dtype=torch.long, device=self.device))


    def update_target_jt(self, reset_env_ids):
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        resample_i[reset_env_ids] = True
        jt_eps_end_bool = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter % self.target_jt_update_steps_int == 0:
            self.target_jt_j += 1
            jt_eps_end_bool = self.target_jt_j >= self.target_jt_seq_len
            resample_i = resample_i | jt_eps_end_bool
            self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)
            self.delayed_obs_target_jt_steps_int = sample_int_from_float(self.delayed_obs_target_jt_steps)
        # if self.cfg.human.resample_on_env_reset:
        self.target_jt_i = torch.where(resample_i, torch.randint(0, self.num_target_jt_seq, (self.num_envs,), device=self.device), self.target_jt_i)

        # TODO(xh) change self.target_jt_seq_len by self.target_jt_i if multi-motion
        if hasattr(self.cfg.init_state, 'random_episode_lenth') and self.cfg.init_state.random_episode_lenth > 0:
            random_init_j = torch.randint(0, self.target_jt_seq_len - self.cfg.init_state.random_episode_lenth, size=self.target_jt_j.shape, device=self.device)
        else:
            random_init_j = torch.zeros(self.target_jt_j.shape, device=self.device, dtype=torch.long)
        self.target_jt_j = torch.where(resample_i, random_init_j, self.target_jt_j)


        self.target_jt_pos, self.target_jt_vel = self.target_jt_pos_seq[self.target_jt_i, self.target_jt_j], self.target_jt_vel_seq[self.target_jt_i, self.target_jt_j]
        # self.target_body_pos, self.target_body_ori, self.target_body_vel, self.target_body_ang_vel = self.target_body_pos_seq[self.target_jt_i, self.target_jt_j], self.target_body_ori_seq[self.target_jt_i, self.target_jt_j], self.target_body_vel_seq[self.target_jt_i, self.target_jt_j], self.target_body_ang_vel_seq[self.target_jt_i, self.target_jt_j]
        if self.cfg.human.load_global:
            self.target_global_pos, self.target_global_ori, self.target_global_vel, self.target_global_ang_vel = self.target_global_pos_seq[self.target_jt_i, self.target_jt_j], self.target_global_ori_seq[self.target_jt_i, self.target_jt_j], self.target_global_vel_seq[self.target_jt_i, self.target_jt_j], self.target_global_ang_vel_seq[self.target_jt_i, self.target_jt_j]
            self.target_heading_quat_inv = diff_rot.calc_heading_quat_inv(vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)))[:,None]

        self.delayed_obs_target_jt_pos = self.target_jt_pos_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]
        self.delayed_obs_target_jt_vel = self.target_jt_vel_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]   

        # self.delayed_obs_target_body_pos = self.target_body_pos_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]
        # self.delayed_obs_target_body_ori = self.target_body_ori_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]  
        # self.delayed_obs_target_body_vel = self.target_body_vel_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]
        # self.delayed_obs_target_body_ang_vel = self.target_body_ang_vel_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]        

        self.delayed_obs_target_global_pos = self.target_global_pos_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]
        self.delayed_obs_target_global_ori = self.target_global_ori_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]  
        self.delayed_obs_target_global_vel = self.target_global_vel_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]
        self.delayed_obs_target_global_ang_vel = self.target_global_ang_vel_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]     
        self.delayed_obs_target_heading_quat_inv = diff_rot.calc_heading_quat_inv(vec6d_to_quat(self.delayed_obs_target_global_ori[:,:6].reshape(-1,3,2)))[:,None]


        # self.delayed_obs_target_root_pos = self.target_root_pos_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]
        # self.delayed_obs_target_root_ori = self.target_root_ori_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]  
        # self.delayed_obs_target_root_vel = self.target_root_vel_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]
        # self.delayed_obs_target_root_ang_vel = self.target_root_ang_vel_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0))]  
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf = self.time_out_buf | jt_eps_end_bool
        self.reset_buf = self.reset_buf | self.time_out_buf
        if 'termination' in self.extras:
            self.extras["termination"]['time_out_buf'] = self.time_out_buf

    # @torch.jit.script
    def update_target_pkls(self, reset_env_ids):
        t0 = time.time()
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        resample_i[reset_env_ids] = True
        jt_eps_end_bool = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter >= self.target_jt_update_steps_int:
            self.common_step_counter = 0
            self.target_jt_j += 1
            jt_eps_end_bool = self.target_jt_j >= self.target_seq_len[self.target_jt_i]
            resample_i = resample_i | jt_eps_end_bool
            self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)
            self.delayed_obs_target_jt_steps_int = sample_int_from_float(self.delayed_obs_target_jt_steps)


        if hasattr(self.cfg, 'recycle_data') and self.cfg.recycle_data:

            obs_global_pos = (self.body_pos_raw - self.env_origins[:,None]).detach().cpu().clone().reshape(self.num_envs,-1)
            obs_global_ori = quat_to_vec6d(self.body_ori_raw).detach().cpu().clone().reshape(self.num_envs,-1)
            obs_global_vel = self.body_vel_raw.detach().cpu().clone().reshape(self.num_envs,-1)
            obs_global_ang_vel = self.body_ang_vel_raw.detach().cpu().clone().reshape(self.num_envs,-1)
            self.extras['obs_global_buf'] = torch.cat([obs_global_pos, obs_global_ori, obs_global_vel, obs_global_ang_vel], dim=1) # NOTE(xh) IMPORTANT BUG!!!!!!!!!!!!
            self.extras['obs_jt_buf'] = torch.cat([self.dof_pos,self.dof_vel], dim=1).detach().cpu().clone()

        # if self.cfg.human.resample_on_env_reset:
        if hasattr(self.cfg,'test') and self.cfg.test:
            self.extras['succ_id'] = self.target_jt_i[jt_eps_end_bool]
            self.extras['dead_id'] = self.target_jt_i[resample_i.nonzero(as_tuple=False).flatten()]
            num_resample = torch.sum(resample_i)
            self.target_jt_i[resample_i.nonzero(as_tuple=False).flatten()] = torch.arange(self.test_motion_id + self.num_envs, 
                                                                                          self.test_motion_id+num_resample + self.num_envs, device=self.device, dtype=torch.long) % self.num_target_seq
            self.test_motion_id+=num_resample
            self.extras['test_motion_id'] = self.test_motion_id
        else:
            self.target_jt_i = torch.where(resample_i, torch.randint(0, self.num_target_seq, (self.num_envs,), device=self.device), self.target_jt_i)

        # TODO(xh) change self.target_jt_seq_len by self.target_jt_i if multi-motion
        if hasattr(self.cfg.init_state, 'random_episode_lenth') and self.cfg.init_state.random_episode_lenth > 0:
            phase = torch.rand(self.target_jt_i.shape, device=self.device)
            random_init_j = (phase * (self.target_seq_len[self.target_jt_i] - self.cfg.init_state.random_episode_lenth)).long()
        else:
            random_init_j = torch.zeros(self.target_jt_j.shape, device=self.device, dtype=torch.long)
        self.target_jt_j = torch.where(resample_i, random_init_j, self.target_jt_j)
        self.episode_length_buf = torch.where(resample_i, 0, self.episode_length_buf)


        target_jt_i_cpu = self.target_jt_i#.to('cpu')
        target_jt_j_cpu = self.target_jt_j#.to('cpu')

        t1 = time.time()
        # self.target_jt = torch.stack([self.target_jt_seq[i][j] for i, j in zip(target_jt_i_cpu, target_jt_j_cpu)], dim=0).to(self.device)
        self.target_jt = self.target_jt_seq[self.motion_start_id[target_jt_i_cpu]+target_jt_j_cpu]#.to(self.device)
        self.target_jt_pos, self.target_jt_vel = self.target_jt[:,:19], self.target_jt[:,19:]

        # self.target_global = torch.stack([self.target_global_seq[i][j] for i, j in zip(target_jt_i_cpu, target_jt_j_cpu)], dim=0).to(self.device)
        self.target_global = self.target_global_seq[self.motion_start_id[target_jt_i_cpu]+target_jt_j_cpu]#.to(self.device)
        self.target_global_pos, self.target_global_ori, self.target_global_vel, self.target_global_ang_vel = \
            self.target_global[:,:3*20], self.target_global[:,3*20:9*20], self.target_global[:,9*20:12*20], self.target_global[:,12*20:15*20]
        self.target_heading_quat_inv = diff_rot.calc_heading_quat_inv(vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)))[:,None]


        t2 = time.time()
        delayed_target_jt_j_cpu = torch.maximum(target_jt_j_cpu - self.delayed_obs_target_jt_steps_int, torch.tensor(0))

        # self.delayed_obs_target_jt = torch.stack([self.target_jt_seq[i][j] for i, j in zip(target_jt_i_cpu, delayed_target_jt_j_cpu)], dim=0).to(self.device)
        self.delayed_obs_target_jt = self.target_jt_seq[self.motion_start_id[target_jt_i_cpu]+delayed_target_jt_j_cpu]#.to(self.device)
        self.delayed_obs_target_jt_pos, self.delayed_obs_target_jt_vel = self.delayed_obs_target_jt[:,:19], self.delayed_obs_target_jt[:,19:]
   
        # self.delayed_obs_target_global = torch.stack([self.target_global_seq[i][j] for i, j in zip(target_jt_i_cpu, delayed_target_jt_j_cpu)], dim=0).to(self.device)
        self.delayed_obs_target_global = self.target_global_seq[self.motion_start_id[target_jt_i_cpu]+delayed_target_jt_j_cpu]#.to(self.device)
        self.delayed_obs_target_global_pos, self.delayed_obs_target_global_ori, self.delayed_obs_target_global_vel, self.delayed_obs_target_global_ang_vel =\
            self.delayed_obs_target_global[:,:3*20], self.delayed_obs_target_global[:,3*20:9*20], self.delayed_obs_target_global[:,9*20:12*20], self.delayed_obs_target_global[:,12*20:15*20]
        self.delayed_obs_target_heading_quat_inv = diff_rot.calc_heading_quat_inv(vec6d_to_quat(self.delayed_obs_target_global_ori[:,:6].reshape(-1,3,2)))[:,None]

   
        # self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf = jt_eps_end_bool
        self.reset_buf = self.reset_buf | self.time_out_buf
        if 'termination' in self.extras:
            self.extras["termination"]['time_out_buf'] = self.time_out_buf
                    
        t3 = time.time()
        # reset_num = int(self.reset_buf.sum()) jt_eps_end_bool.nonzero(as_tuple=False).flatten()
        # print(f'{t1-t0:.2f}s')
        # print(f'{t2-t1:.2f}s', reset_num)  self.episode_length_buf[env_ids]
        # print(f'{t3-t2:.2f}s')
        # print('\n')

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # step physics and render each frame
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.render()
        if self.action_delay != -1:
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], actions[:, None, :]], dim=1)
            actions = self.action_history_buf[:, -self.action_delay - 1] # delay for 1/50=20ms
        self.actions = actions.clone()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # breakpoint()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        if self.cfg.normalization.clip_observations:
            clip_obs = self.cfg.normalization.clip_observations
            self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # clip_obs = self.cfg.normalization.clip_observations
        # self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # if self.privileged_obs_buf is not None:
        #     self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.env.policy_name == "ActorCritic":
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        else:    
            return self.obs_history_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim) # change self.root_states simutanously
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.heading_quat[:] = diff_rot.calc_heading_quat(self.base_quat)[:,None]
        self.heading_quat_inv[:] = diff_rot.calc_heading_quat_inv(self.base_quat)[:,None]
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_lin_vel[:] = quat_rotate_inverse(self.heading_quat[:,0], self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.heading_quat[:,0], self.root_states[:, 10:13])
        self.base_orn_rp[:] = self.get_body_orientation()
        # self.base_ori_inv = quat_inv(self.base_quat)

        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.root_pos = self.root_states[:, :3]
        self.root_ori = self.root_states[:, 3:7]
        self.root_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:]

        # self.pelvis_pos = self.body_pos_raw[:, :1, :]
        # self.pelvis_ori = self.body_ori_raw[:, :1, :]
        # self.pelvis_ori_inv = quat_inv(self.pelvis_ori) # same as base_ori_inv
        self.body_pos = broadcast_quat_apply(self.heading_quat_inv, self.body_pos_raw - self.root_pos[:,None]).reshape(-1, 3 * (self.num_dofs + 1))
        self.body_ori = flip_quat_by_w(broadcast_quat_multiply(self.heading_quat_inv, self.body_ori_raw))
        self.body_ori = quat_to_vec6d(self.body_ori).reshape(-1, 6 * (self.num_dofs + 1))
        self.body_vel = broadcast_quat_apply(self.heading_quat_inv, self.body_vel_raw).reshape(-1, 3 * (self.num_dofs + 1))
        self.body_ang_vel = broadcast_quat_apply(self.heading_quat_inv, self.body_ang_vel_raw).reshape(-1, 3 * (self.num_dofs + 1))

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination() # self.reset_buf = termination_contact_buf | r_threshold_buff | p_threshold_buff | z_threshold_buff | self.time_out_buf
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if hasattr(self.cfg.human, 'multi_motion') and self.cfg.human.multi_motion:
            self.update_target_pkls(env_ids)
        else:
            self.update_target_jt(env_ids)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids) # Will body info updated here?
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.heading_quat[:] = diff_rot.calc_heading_quat(self.base_quat)[:,None]
        self.heading_quat_inv[:] = diff_rot.calc_heading_quat_inv(self.base_quat)[:,None]
        self.body_pos = broadcast_quat_apply(self.heading_quat_inv, self.body_pos_raw - self.root_pos[:,None]).reshape(-1, 3 * (self.num_dofs + 1))
        self.body_ori = flip_quat_by_w(broadcast_quat_multiply(self.heading_quat_inv, self.body_ori_raw))
        self.body_ori = quat_to_vec6d(self.body_ori).reshape(-1, 6 * (self.num_dofs + 1))
        self.body_vel = broadcast_quat_apply(self.heading_quat_inv, self.body_vel_raw).reshape(-1, 3 * (self.num_dofs + 1))
        self.body_ang_vel = broadcast_quat_apply(self.heading_quat_inv, self.body_ang_vel_raw).reshape(-1, 3 * (self.num_dofs + 1))

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        

    def check_termination(self):
        """ Check if environments need to be reset
        """

        if (hasattr(self.cfg,'test') and self.cfg.test) or (hasattr(self.cfg,'play') and self.cfg.play):
            contact_threshold = 9999999999999
            r_threshold = 9999999999999
            p_threshold = 9999999999999
            z_threshold = 0
            termination_threshold = self.cfg.termination.termination_distance
        else:
            contact_threshold = self.cfg.termination.contact_threshold
            r_threshold = self.cfg.termination.r_threshold * self.scaling_factor * 10
            p_threshold = self.cfg.termination.p_threshold * self.scaling_factor * 10
            z_threshold = self.cfg.termination.z_threshold / (self.scaling_factor * 10)
            termination_threshold = self.cfg.termination.termination_distance / self.scaling_factor

        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > contact_threshold, dim=1)

        r, p = self.base_orn_rp[:, 0], self.base_orn_rp[:, 1]
        z = self.root_states[:, 2]


        r_threshold_buff = r.abs() > r_threshold
        p_threshold_buff = p.abs() > p_threshold
        z_threshold_buff = z < z_threshold
        has_fallen = torch.zeros_like(r_threshold_buff)


        if hasattr(self.cfg.termination,'enable_early_termination') and self.cfg.termination.enable_early_termination:
            if self.cfg.env.obs.target_global_pos:
                global_body_pos = (self.body_pos_raw - self.env_origins[:,None])
                global_body_pos[...,2] = global_body_pos[...,2] - global_body_pos[...,0:1,2]
                global_body_pos_znorm=global_body_pos
                target_global_pos=self.target_global_pos.view(-1,20,3)
                target_global_pos[...,2] = target_global_pos[...,2] - target_global_pos[...,0:1,2]
                target_global_pos_znorm=target_global_pos
                # global_body_pos = (self.body_pos_raw - self.env_origins[:,None])
                # mean_global_distance = torch.mean(torch.norm(global_body_pos - self.target_global_pos.view(-1,self.num_dofs + 1,3), dim=-1), dim=-1)
                mean_global_distance = torch.mean(torch.norm(global_body_pos_znorm - target_global_pos_znorm, dim=-1), dim=-1)
                has_fallen = torch.any(torch.norm(global_body_pos - self.target_global_pos.view(-1,self.num_dofs + 1,3), dim=-1) > termination_threshold, dim=-1)  # using max
                self.extras['mean_global_distance'] = mean_global_distance
            elif self.cfg.env.obs.target_body_pos:
                local_body_pos = self.body_pos.reshape(-1,self.num_dofs + 1,3)
                target_local_body_pos = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                            (self.target_global_pos.reshape(-1,self.num_dofs + 1,3) - 
                                                            self.target_global_pos[:,None,:3]))
                has_fallen = torch.any(torch.norm(local_body_pos - target_local_body_pos, dim=-1) > termination_threshold, dim=-1)  # using max
            else:
                NotImplementedError
            has_fallen = has_fallen & (self.episode_length_buf > 10)
            self.reset_buf = termination_contact_buf | r_threshold_buff | p_threshold_buff | z_threshold_buff | has_fallen #| self.time_out_buf #check in update_target_jt
        else:
            self.reset_buf = termination_contact_buf | r_threshold_buff | p_threshold_buff | z_threshold_buff #| self.time_out_buf
        self.extras["termination"] = {}
        self.extras["termination"]['termination_contact_buf'] = termination_contact_buf
        self.extras["termination"]['r_threshold_buff'] = r_threshold_buff
        self.extras["termination"]['p_threshold_buff'] = p_threshold_buff
        self.extras["termination"]['z_threshold_buff'] = z_threshold_buff
        self.extras["termination"]['has_fallen'] = has_fallen

        # self.extras["termination"]['time_out_buf'] = self.time_out_buf


    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        # print(self.target_jt_j[env_ids])
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.simulate(self.sim)
        self._reset_root_states(env_ids)
        self._reset_dofs(env_ids)
        # self.gym.simulate(self.sim) # NOTE(xh)

        
        # self.body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        # self.root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        # self.root_states +=1
        # # root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)  
        # self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.root_states))
        # self.gym.refresh_actor_root_state_tensor(self.sim,)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))

        # self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        if self.episode_length_buf.float().mean().item() < 100:
            self.scaling_factor = max(self.scaling_factor * 0.999, .1)
        elif self.episode_length_buf.float().mean().item()  > (200 if self.cfg.human.multi_motion else 600):
            self.scaling_factor = min(self.scaling_factor * 1.001, 1.)
        self.extras['episode_length_current'] = self.episode_length_buf.float().mean()
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode_metrics"] = deepcopy(self.episode_metrics)
        self.extras['scaling_factor'] = self.scaling_factor
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            unscaled_rew, metric = self.reward_functions[i]()
            if self.reward_scales[name] < 0:
                unscaled_rew *= self.scaling_factor
            rew = unscaled_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metrics[name] = metric.mean().item()
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
            
        B, _ = self.dof_pos.shape
        
        obs_list = []
        # TODO(xh) body: root frame, local: heading frame, global: world frame

        ############### state ################

        if self.cfg.env.obs.dof_pos: # num_dofs local
            obs_list.append(self.dof_pos * self.obs_scales.dof_pos)
        if self.cfg.env.obs.dof_vel: # num_dofs
            obs_list.append(self.dof_vel * self.obs_scales.dof_vel)

        if self.cfg.env.obs.body_pos: # 3 * (num_dofs + 1) wtf heading frame
            obs_list.append(self.body_pos * self.obs_scales.body_pos)
        if self.cfg.env.obs.body_ori: # 6 * (num_dofs + 1) wtf heading frame
            obs_list.append(self.body_ori * self.obs_scales.body_ori)
        if self.cfg.env.obs.body_vel: # 3 * (num_dofs + 1) wtf heading frame
            obs_list.append(self.body_vel * self.obs_scales.body_vel)
        if self.cfg.env.obs.body_ang_vel: # 3 * (num_dofs + 1) wtf heading frame
            obs_list.append(self.body_ang_vel * self.obs_scales.body_ang_vel) 

        if self.cfg.env.obs.root_pos: # 3 global w/o env origins
            obs_list.append((self.root_pos-self.env_origins) * self.obs_scales.root_pos)
        if self.cfg.env.obs.root_ori: # 6 global
            obs_list.append(quat_to_vec6d(self.root_ori).reshape([-1,6]) * self.obs_scales.root_ori)
        if self.cfg.env.obs.root_vel: # 3 global
            obs_list.append(self.root_vel * self.obs_scales.root_vel)
        if self.cfg.env.obs.root_ang_vel: # 3 global
            obs_list.append(self.root_ang_vel * self.obs_scales.root_ang_vel)  


        if self.cfg.env.obs.root_high: # 1
            obs_list.append((self.root_pos-self.env_origins)[...,-1:] * self.obs_scales.root_pos)

        if self.cfg.env.obs.last_action: # num_dofs
            obs_list.append(self.actions * self.obs_scales.last_action)

        if self.cfg.env.obs.base_orn_rp: # 2
            obs_list.append(self.base_orn_rp * self.obs_scales.base_orn_rp)
        if self.cfg.env.obs.base_lin_vel: # 3 wtf heading frame
            obs_list.append(self.base_lin_vel * self.obs_scales.base_lin_vel) 
        if self.cfg.env.obs.base_ang_vel: # 3 wtf heading frame
            obs_list.append(self.base_ang_vel * self.obs_scales.base_ang_vel) 
        if self.cfg.env.obs.projected_gravity: # wtf root frame
            obs_list.append(self.projected_gravity * self.obs_scales.base_ang_vel)

        if self.cfg.env.obs.commands: # 3
            obs_list.append(self.commands[:, :3] * self.commands_scale[:3])  

        ##################### task #####################

        if self.cfg.env.obs.target_dof_pos: # num_dofs
            obs_list.append(self.delayed_obs_target_jt_pos * self.obs_scales.target_dof_pos)
        if self.cfg.env.obs.target_dof_vel: # num_dofs
            obs_list.append(self.delayed_obs_target_jt_vel * self.obs_scales.target_dof_vel)

        if self.cfg.env.obs.target_body_pos: # 3 * (num_dofs + 1) wtf heading frame
            target_body_pos = self.delayed_obs_target_global_pos.reshape(-1, self.num_dofs + 1, 3)  - self.delayed_obs_target_global_pos[:,None,:3]
            target_body_pos = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv, target_body_pos).reshape(-1, 3 * (self.num_dofs + 1))
            obs_list.append(target_body_pos * self.obs_scales.target_body_pos)
        if self.cfg.env.obs.target_body_ori: # 6 * (num_dofs + 1) wtf heading frame
            target_body_ori = broadcast_quat_multiply(self.delayed_obs_target_heading_quat_inv,
                vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(B,self.num_dofs + 1,3,2)))
            target_body_ori = quat_to_vec6d(target_body_ori).reshape(-1, 6 * (self.num_dofs + 1))
            obs_list.append(target_body_ori * self.obs_scales.target_body_ori)
        if self.cfg.env.obs.target_body_vel: # 3 * (num_dofs + 1) none
            obs_list.append(self.delayed_obs_target_body_vel * self.obs_scales.target_body_vel)
        if self.cfg.env.obs.target_body_ang_vel: # 3 * (num_dofs + 1) none
            obs_list.append(self.delayed_obs_target_body_ang_vel * self.obs_scales.target_body_ang_vel) 

        if self.cfg.env.obs.target_global_pos: # 3 * (num_dofs + 1) wtf heading frame
            target_global_pos = self.delayed_obs_target_global_pos.reshape(-1, self.num_dofs + 1, 3) - (self.root_pos-self.env_origins)[:,None]
            target_global_pos = broadcast_quat_apply(self.heading_quat_inv, target_global_pos).reshape(-1, 3 * (self.num_dofs + 1))
            obs_list.append(target_global_pos * self.obs_scales.target_global_pos)
        if self.cfg.env.obs.target_global_ori: # 6 * (num_dofs + 1) wtf heading frame
            target_global_ori = broadcast_quat_multiply(self.heading_quat_inv,
                vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(B,self.num_dofs + 1,3,2)))
            target_global_ori = quat_to_vec6d(target_global_ori).reshape(-1, 6 * (self.num_dofs + 1))
            obs_list.append(target_global_ori * self.obs_scales.target_global_ori)
        if self.cfg.env.obs.target_global_vel: # 3 * (num_dofs + 1)
            obs_list.append(self.delayed_obs_target_global_vel * self.obs_scales.target_global_vel)
        if self.cfg.env.obs.target_global_ang_vel: # 3 * (num_dofs + 1)
            obs_list.append(self.delayed_obs_target_global_ang_vel * self.obs_scales.target_global_ang_vel) 



        if self.cfg.env.obs.diff_local_pos: # 3 * (num_dofs + 1)
            target_body_pos = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv,
                (self.delayed_obs_target_global_pos.reshape(-1, self.num_dofs + 1, 3) - self.delayed_obs_target_global_pos[:,None,:3]))
            diff_local_pos = (target_body_pos.reshape(-1, 3 * (self.num_dofs + 1)) - self.body_pos)
            obs_list.append(diff_local_pos * self.obs_scales.diff_local_pos)
        if self.cfg.env.obs.diff_local_ori: # 6 * (num_dofs + 1) 
            local_body_ori = broadcast_quat_multiply(self.heading_quat_inv, 
                                                 self.body_ori_raw)
            target_local_body_ori = broadcast_quat_multiply(self.delayed_obs_target_heading_quat_inv, 
                                                            vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(-1,self.num_dofs + 1,3,2)).reshape(-1,20,4))
            diff_local_body_rot = broadcast_quat_multiply(target_local_body_ori, 
                diff_rot.quat_conjugate(local_body_ori))
            diff_local_body_rot = quat_to_vec6d(diff_local_body_rot).reshape(B, 6 * (self.num_dofs + 1))
            obs_list.append(diff_local_body_rot * self.obs_scales.diff_local_ori)
        if self.cfg.env.obs.diff_local_vel: # 3 * (num_dofs + 1)
            target_local_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv, 
                                                self.delayed_obs_target_global_vel.reshape(-1,self.num_dofs + 1,3)).reshape(-1, 3 * (self.num_dofs + 1))
            diff_local_vel = target_local_vel - self.body_vel
            obs_list.append(diff_local_vel * self.obs_scales.diff_local_vel)
        if self.cfg.env.obs.diff_local_ang_vel: # 3 * (num_dofs + 1)
            target_local_ang_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv, 
                                                self.delayed_obs_target_global_ang_vel.reshape(-1,self.num_dofs + 1,3)).reshape(-1, 3 * (self.num_dofs + 1))
            diff_local_ang_vel = target_local_ang_vel - self.body_ang_vel
            obs_list.append(diff_local_ang_vel * self.obs_scales.diff_local_ang_vel)

        if self.cfg.env.obs.diff_global_pos: # 3 * (num_dofs + 1)
            diff_global_pos = self.delayed_obs_target_global_pos.reshape(B, self.num_dofs + 1, 3) - (self.body_pos_raw - self.env_origins[:,None])
            diff_global_pos = broadcast_quat_apply(self.heading_quat_inv, diff_global_pos).reshape(-1, 3 * (self.num_dofs + 1))
            obs_list.append(diff_global_pos * self.obs_scales.diff_global_pos)
        if self.cfg.env.obs.diff_global_ori: # 6 * (num_dofs + 1) 
            diff_global_body_rot = broadcast_quat_multiply(vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(B,self.num_dofs + 1,3,2)), 
                                    diff_rot.quat_conjugate(self.body_ori_raw))
            diff_global_body_rot = broadcast_quat_multiply(self.heading_quat_inv,
                                    broadcast_quat_multiply(diff_global_body_rot, 
                                    self.heading_quat))
            diff_global_body_rot = quat_to_vec6d(diff_global_body_rot).reshape(B, 6 * (self.num_dofs + 1))
            obs_list.append(diff_global_body_rot * self.obs_scales.diff_global_ori)
        if self.cfg.env.obs.diff_global_vel: # 3 * (num_dofs + 1)
            diff_global_vel = self.delayed_obs_target_global_vel.reshape(B, self.num_dofs + 1, 3) - self.body_vel_raw
            diff_global_vel = broadcast_quat_apply(self.heading_quat_inv, diff_global_vel).reshape(-1, 3 * (self.num_dofs + 1))
            obs_list.append(diff_global_vel * self.obs_scales.diff_global_vel)
        if self.cfg.env.obs.diff_global_ang_vel: # 3 * (num_dofs + 1)
            diff_global_ang_vel = self.delayed_obs_target_global_ang_vel.reshape(B, self.num_dofs + 1, 3) - self.body_ang_vel_raw
            diff_global_ang_vel = broadcast_quat_apply(self.heading_quat_inv, diff_global_ang_vel).reshape(-1, 3 * (self.num_dofs + 1))
            obs_list.append(diff_global_ang_vel * self.obs_scales.diff_global_ang_vel)


        if self.cfg.env.obs.target_root_pos: # 3 global
            obs_list.append(self.delayed_obs_target_global_pos[:,:3] * self.obs_scales.target_root_pos)
        if self.cfg.env.obs.target_root_ori: # 6 global
            obs_list.append(self.delayed_obs_target_global_ori[:,:6] * self.obs_scales.target_root_ori)
        if self.cfg.env.obs.target_root_vel: # 3 local
            target_root_local_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv.squeeze(1), 
                                                self.delayed_obs_target_global_vel[:,:3])
            diff_root_local_vel = target_root_local_vel - self.base_lin_vel
            obs_list.append(diff_root_local_vel * self.obs_scales.target_root_vel)
            # obs_list.append(self.delayed_obs_target_body_vel[:,:3] * self.obs_scales.target_root_vel)
        if self.cfg.env.obs.target_root_ang_vel: # 3 local
            target_root_local_ang_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv.squeeze(1), 
                                                self.delayed_obs_target_global_ang_vel[:,:3])
            diff_root_local_ang_vel = target_root_local_ang_vel - self.base_ang_vel
            obs_list.append(diff_root_local_ang_vel * self.obs_scales.target_root_ang_vel) 
            # obs_list.append(self.delayed_obs_target_body_ang_vel[:,:3] * self.obs_scales.target_root_ang_vel) 

         
        self.obs_buf = torch.cat(obs_list, dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.cfg.noise.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
        self.obs_history_buf = torch.cat([ # TODO(xh)
            self.obs_history_buf[:, 1:],
            self.obs_buf.unsqueeze(1)
        ], dim=1)

    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        if return_yaw:
            return torch.stack([r, p, y], dim=-1)
        else:
            return torch.stack([r, p], dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            target_dof_pos = actions_scaled + self.default_dof_pos #NOTE(xh)
            # if self.cfg.control.clip_actions:
            #     target_dof_pos = torch.clip(target_dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
            torques = self.p_gains*(target_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # NOTE(xh) init from target pose, only for debug
        dof_state_copy = self.dof_state.clone().view(self.num_envs, self.num_dofs, 2)
        if hasattr(self.cfg.env, 'target_init') and self.cfg.env.target_init:
            dof_state_copy[..., 0][env_ids] = self.target_jt_pos[env_ids] # NOTE(xh) self.target_jt_pos should update bufore reset dof
        else:
            dof_state_copy[..., 0][env_ids] = self.default_dof_pos * torch_rand_float(0.9, 1.1, (len(env_ids), self.num_dofs), device=self.device)
        dof_state_copy[..., 1][env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_dof_state_tensor(self.sim,
        #                                       gymtorch.unwrap_tensor(dof_state_copy.view(-1,2)))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(dof_state_copy.view(-1,2)),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        del dof_state_copy
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_states_copy = self.root_states.clone()
        if not (hasattr(self.cfg.init_state, 'random_episode_lenth') and self.cfg.init_state.random_episode_lenth > 0):
            assert (self.target_jt_j[env_ids] < 2).all()
        if hasattr(self.cfg.env, 'target_init') and self.cfg.env.target_init:
            root_states_copy[env_ids, :3] = self.target_global_pos[env_ids,:3]
            root_states_copy[env_ids, :3] += self.env_origins[env_ids]
            root_states_copy[env_ids,3:7] = vec6d_to_quat(self.target_global_ori[env_ids,:6].reshape(-1,3,2))
        elif hasattr(self.cfg.env, 'target_heading_init') and self.cfg.env.target_heading_init:
            root_states_copy[env_ids, :3] = self.target_global_pos[env_ids,:3]
            root_states_copy[env_ids, :3] += self.env_origins[env_ids]
            root_states_copy[env_ids,3:7] = diff_rot.calc_heading_quat(vec6d_to_quat(self.target_global_ori[env_ids,:6].reshape(-1,3,2)))
        elif self.custom_origins:
            root_states_copy[env_ids] = self.base_init_state
            root_states_copy[env_ids, :3] += self.env_origins[env_ids]
            root_states_copy[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            root_states_copy[env_ids] = self.base_init_state
            root_states_copy[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        root_states_copy[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_actor_root_state_tensor(self.sim,
        #                                              gymtorch.unwrap_tensor(root_states_copy))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(root_states_copy),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        del root_states_copy
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # gym = self.gym
        # sim = self.sim
        # gym.step_graphics(sim)
        # gym.draw_viewer(self.viewer, sim, True)
        # gym.clear_lines(self.viewer)
        # gym.sync_frame_time(sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:2] = noise_scales.base_orn * noise_level * self.obs_scales.base_orn_rp
        noise_vec[2:5] = noise_scales.base_ang_vel * noise_level * self.obs_scales.base_ang_vel
        noise_vec[5:8] = 0. # commands
        noise_vec[8: 8 + self.num_dofs] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[8 + self.num_dofs: 8 + 2 * self.num_dofs] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[8 + 2 * self.num_dofs: 8 + 3 * self.num_dofs] = 0 # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[-self.terrain.num_height_points:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors 
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim) # (2048, 13)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # 
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.body_state = gymtorch.wrap_tensor(body_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.body_pos_raw = self.body_state.view(self.num_envs, self.num_dofs+1, -1)[..., :3]
        self.body_ori_raw = self.body_state.view(self.num_envs, self.num_dofs+1, -1)[..., 3:7]
        self.body_vel_raw = self.body_state.view(self.num_envs, self.num_dofs+1, -1)[..., 7:10]
        self.body_ang_vel_raw = self.body_state.view(self.num_envs, self.num_dofs+1, -1)[..., 10:]

        self.base_quat = self.root_states[:, 3:7]
        
        self.root_pos = self.root_states[:, :3]
        self.root_ori = self.root_states[:, 3:7]
        self.root_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        if self.cfg.noise.add_noise:
            self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(self.cfg.normalization.commands_scale, device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.heading_quat = diff_rot.calc_heading_quat(self.base_quat)[:,None]
        self.heading_quat_inv = diff_rot.calc_heading_quat_inv(self.base_quat)[:,None]
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_lin_vel = quat_rotate_inverse(self.heading_quat[:,0], self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.heading_quat[:,0], self.root_states[:, 10:13])
        self.base_orn_rp = self.get_body_orientation() # [r, p]
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        assert(np.all([name1 == name2 for name1, name2 in zip(self.dof_names, self.cfg.init_state.default_joint_angles.keys())]))
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        # current non-zero reward: tracking_lin_vel, tracking_ang_vel, target_jt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.episode_metrics = {name: 0 for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
     
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        assert(self.num_bodies == len(body_names))
        assert(self.num_dofs == len(self.dof_names))
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        # if hasattr(self.cfg,'play') and self.cfg.play:

        #     asset_options = gymapi.AssetOptions()
        #     asset_options.angular_damping = 0.0
        #     asset_options.linear_damping = 0.0
        #     asset_options.max_angular_velocity = 0.0
        #     asset_options.density = 0
        #     asset_options.fix_base_link = True
        #     asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        #     self._marker_asset = self.gym.load_asset(self.sim, asset_root, "traj_marker.urdf", asset_options)
        #     default_pose = gymapi.Transform()
        #     self._marker_handles=[]
        #     for i in range(self.num_dofs):
        #         marker_handle = self.gym.create_actor(env_handle, self._marker_asset, default_pose, "marker", self.num_envs + 10, 1, 0)
        #         self.gym.set_rigid_body_color(env_handle, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0))
        #         self._marker_handles.append(marker_handle)
        #     self.env_handle_ref = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
        #     rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
        #     self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
        #     self.actor_handle_ref = self.gym.create_actor(self.env_handle_ref, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
        #     dof_props = self._process_dof_props(dof_props_asset, i)
        #     self.gym.set_actor_dof_properties(self.env_handle_ref, self.actor_handle_ref, dof_props)
        #     body_props = self.gym.get_actor_rigid_body_properties(self.env_handle_ref, self.actor_handle_ref)
        #     body_props = self._process_rigid_body_props(body_props, i)
        #     self.gym.set_actor_rigid_body_properties(self.env_handle_ref, self.actor_handle_ref, body_props, recomputeInertia=True)
        #     for t in range(20):
        #         self.gym.set_rigid_body_color(self.env_handle_ref,self.actor_handle_ref,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.6,0,0))

       
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False) # TODO(xh)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self.dof_hip_indices = [i for i, name in enumerate(self.dof_names) if 'hip' in name]
        self.dof_hip_indices = torch.tensor(self.dof_hip_indices, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.dof_hip_yaw_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'hip_yaw' in name], dtype=torch.long, device=self.device, requires_grad=False)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        print('penalized_contact_indices: {}'.format(self.penalized_contact_indices))
        print('termination_contact_indices: {}'.format(self.termination_contact_indices))
        print('feet_indices: {}'.format(self.feet_indices))

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.action_delay = self.cfg.env.action_delay

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        # if not self.terrain.cfg.measure_heights:
        #     return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            # base_pos = (self.root_states[i, :3]).cpu().numpy()
            # heights = self.measured_heights[i].cpu().numpy()
            # height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            # for j in range(heights.shape[0]):
            #     x = height_points[j, 0] + base_pos[0]
            #     y = height_points[j, 1] + base_pos[1]
            #     z = heights[j]
            #     sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            for j in range(self.num_dofs+1):
                if j in self.termination_contact_indices:
                    pos = self.target_global_pos[i,3*j:3*j+3] + self.env_origins[i]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)

                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 


    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    #------------ reward functions----------------
    def _reward_jt_base_vel_xy_humanplus(self):
        # Track target base vel xy
        pass # not implemented
        base_vel_xy_error = torch.sum(torch.abs(self.base_lin_vel[:, :2] - self.target_jt_base_lin_vel[:, :2]))
        return torch.exp(-base_vel_xy_error), base_vel_xy_error

    def _reward_jt_base_ang_vel_yaw_humanplus(self):
        # Track target base ang vel yaw
        pass # not implemented
        base_ang_vel_yaw_error = torch.abs(self.base_ang_vel[:, 2] - self.target_jt_base_ang_vel[:, 2])
        return torch.exp(-base_ang_vel_yaw_error), base_ang_vel_yaw_error
    
    def _reward_jt_dof_pos_humanplus(self):
        # Track target dof pos
        target_jt_pos = self.target_jt_pos * 0#+self.default_dof_pos 
        target_jt_pos_error = torch.sum(torch.square(self.dof_pos - target_jt_pos), dim=1)
        return target_jt_pos_error, target_jt_pos_error
        
    def _reward_jt_base_rp_humanplus(self):
        # Track target base rp
        pass # not implemented
        base_rp_error = torch.sum(torch.square(self.base_orn_rp - self.target_jt_base_orn_rp))
        return base_rp_error, base_rp_error
    
    def _reward_energy_humanplus(self):
        # Regularize energy
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1), torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_jt_foot_contact_humanplus(self):
        # Track target foot contact
        pass

    def _reward_feet_slipping_humanplus(self):
        # Penalize feet slipping
        return torch.norm((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1, p=2) > 1) * torch.norm(self.body_vel_raw[:, self.feet_indices, :], dim=-1, p=2), dim=-1, p=2), torch.norm((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1, p=2) > 1) * torch.norm(self.body_vel_raw[:, self.feet_indices, :], dim=-1, p=2), dim=-1, p=2)

    def _reward_alive_humanplus(self):
        # Alive reward
        # return torch.ones(self.num_envs), torch.ones(self.num_envs)
        return 1.-1.*self.reset_buf, 1.-1.*self.reset_buf
    
    def _reward_tracking_lin_vel_jaylon(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma), torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_jaylon(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma), torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_orientation_jaylon(self):
        # Penalize non-flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1), torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_energy_jaylon(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
        return energy, energy

    def _reward_dof_vel_jaylon(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1), torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc_jaylon(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1), torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_weighted_torques_jaylon(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques / self.p_gains.view(1, -1)), dim=1), torch.sum(torch.square(self.torques / self.p_gains.view(1, -1)), dim=1)

    def _reward_contact_forces_jaylon(self):
        contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        waste_force = contact_forces - self.cfg.rewards.max_contact_force
        rew = torch.sum(torch.clip(waste_force, 0., 500), dim=1)  # exceed 500
        return rew, rew
    
    def _reward_collision_jaylon(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1), torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate_jaylon(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1), torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_hip_yaw_dof_error_jaylon(self):
        # Penalize arm dof error
        hip_diff = self.dof_pos[:, self.dof_hip_yaw_indices] - self.default_dof_pos[:, self.dof_hip_yaw_indices]
        return torch.sum(torch.square(hip_diff), dim=1), torch.sum(torch.square(hip_diff), dim=1)

    def _reward_feet_away_jaylon(self):
        # Penalize feet away from the ground
        feet_threshold = 0.4
        feet_0_pos = self.body_pos_raw[:, self.feet_indices[0], :3]
        feet_1_pos = self.body_pos_raw[:, self.feet_indices[1], :3]
        feet_distance = torch.norm(feet_0_pos - feet_1_pos, dim=1)
        mask = feet_distance > feet_threshold
        rew = mask * feet_threshold + ~mask * feet_distance
        return rew, rew

    def _reward_stance_jaylon(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        any_contact = torch.any(contact, dim=1).to(torch.float32)  # contact on z axis
        return any_contact, any_contact

    def _reward_stand_still_jaylon(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1), torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]), torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1), torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1), torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target), torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1), torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_weighted_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques / self.p_gains.view(1, -1)), dim=1), torch.sum(torch.square(self.torques / self.p_gains.view(1, -1)), dim=1)
    
    def _reward_actions(self):
        # Penalize torques
        return torch.sum(torch.square(self.actions), dim=1), torch.sum(torch.square(self.actions), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1), torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        dof_acc = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        dof_acc[self.episode_length_buf <= 5] = 0
        return dof_acc, dof_acc
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        action_rate=torch.square(self.last_actions - self.actions)
        action_rate = torch.sum(action_rate, dim=1)
        action_rate[self.episode_length_buf <= 5] = 0
        return action_rate, action_rate
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 0.1), dim=1), torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf, self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1), torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1), torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1), torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma), lin_vel_error
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma), ang_vel_error

    def _reward_feet_air_time(self): # TODO: check if this is correct
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime, rew_airTime
    
    def _reward_feet_air(self): # TODO: check if this is correct
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] < 1.
        contact_air = torch.logical_and(contact[:,0], contact[:,1]) 
        # self.last_contacts = contact
        # first_contact = (self.feet_air_time > 0.) * contact_filt
        # self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # self.feet_air_time *= ~contact_filt
        return contact_air, contact_air
    
    def _reward_feet_away(self):
        # Penalize feet away from the ground
        feet_threshold = 0.4
        feet_0_pos = self.rigid_body_state[:, self.feet_indices[0], :3]
        feet_1_pos = self.rigid_body_state[:, self.feet_indices[1], :3]
        feet_distance = torch.norm(feet_0_pos - feet_1_pos, dim=1)
        mask = feet_distance > feet_threshold
        rew = mask * feet_threshold + ~mask * feet_distance
        return rew, rew
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1), torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    
    def _reward_stance(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        any_contact = torch.any(contact, dim=1).to(torch.float32)  # contact on z axis
        return any_contact, any_contact
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1), torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1), torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_feet_slipping(self):
        # penalize high contact forces
        return torch.norm((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1) * torch.norm(self.body_cart_vel[:, self.feet_indices, :], dim=-1), dim=-1), torch.norm((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1) * torch.norm(self.body_cart_vel[:, self.feet_indices, :], dim=-1), dim=-1)

    def _reward_target_jt_pos(self):
        # Penalize distance to target joint angles
        target_jt_pos = self.target_jt_pos#+self.default_dof_pos # NOTE(xh): add default pos to target jt
        target_jt_pos_error = torch.mean(torch.abs(self.dof_pos - target_jt_pos), dim=1)
        return torch.exp(-4 * target_jt_pos_error), target_jt_pos_error
    
    def _reward_target_jt_vel(self):
        # Penalize velocity distance to target joint angles
        target_jt_vel_error = torch.mean(torch.abs(self.dof_vel - self.target_jt_vel), dim=1) 
        return torch.exp(-4 * target_jt_vel_error), target_jt_vel_error
    
    def _reward_target_jt_cart_pos(self):
        pass
    
    def _reward_target_jt_cart_vel(self):
        pass

    def _reward_energy(self):
        # Penalize energy cost
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1), torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
    
    def _reward_gp_MaskedMimic(self):
        # breakpoint()
        target_global_pos = self.target_global_pos.view(self.body_pos_raw.shape) + self.env_origins[:,None]
        target_gp_error = (torch.sum(torch.square(target_global_pos - self.body_pos_raw), dim=1)).mean(dim=1)
        return torch.exp(-0.2 * target_gp_error), target_gp_error
    
    def _reward_gr_MaskedMimic(self):
        # breakpoint()
        global_ori = quat_to_vec6d(flip_quat_by_w(self.body_ori_raw), do_normalize=True).reshape(-1,20,6)
        target_gr_error = (torch.sum(torch.square(self.target_global_ori.reshape(-1,20,6) - global_ori), dim=1)).mean(dim=1)
        return torch.exp(-0.2 * target_gr_error), target_gr_error
    
    def _reward_rh_MaskedMimic(self):
        # breakpoint()
        target_rh_error = torch.square(self.target_global_pos[:, 2] - self.body_pos_raw[:, 0, -1])
        return torch.exp(-1. * target_rh_error), target_rh_error
    
    def _reward_jv_MaskedMimic(self):
        # breakpoint()
        target_jv_error = (torch.sum(torch.square(self.target_global_vel.reshape(-1,20,3) - self.body_vel_raw), dim=1)).mean(dim=1)
        return torch.exp(-0.08 * target_jv_error), target_jv_error
    
    def _reward_jav_MaskedMimic(self):
        # breakpoint()
        target_jav_error = (torch.sum(torch.square(self.target_global_ang_vel.reshape(-1,20,3) - self.body_ang_vel_raw), dim=1)).mean(dim=1)
        return torch.exp(-0.03 * target_jav_error), target_jav_error
    
    def _reward_eg_MaskedMimic(self):
        # breakpoint()
        eg_error = torch.sum(torch.square(self.torques), dim=1)
        return -eg_error, eg_error
    
    def _reward_root_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # target_root_v = quat_rotate_inverse(vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)), self.target_global_vel[:,:3])
        target_root_local_vel = broadcast_quat_apply(self.target_heading_quat_inv.squeeze(1), 
                                            self.target_global_vel[:,:3])
        diff_root_local_vel = target_root_local_vel - self.base_lin_vel
        # target_body_vel ??TODO(xh)
        lin_vel_error = torch.sum(torch.square(diff_root_local_vel), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma), lin_vel_error
    
    def _reward_root_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        # target_root_av = quat_rotate_inverse(vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)), self.target_global_ang_vel[:,:3])
        # target_body_ang_vel??
        target_root_local_ang_vel = broadcast_quat_apply(self.target_heading_quat_inv.squeeze(1), 
                                            self.target_global_ang_vel[:,:3])
        diff_root_local_ang_vel = target_root_local_ang_vel - self.base_ang_vel
        ang_vel_error = torch.sum(torch.square(diff_root_local_ang_vel), dim=1)
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma), ang_vel_error
        
    def _reward_root_ori(self):
        # Tracking of linear velocity commands (xy axes)
        
        target_root_ori = self.target_global_ori[:,:6]
        root_ori_error = torch.sum(torch.square(target_root_ori - quat_to_vec6d(self.root_ori).reshape(-1,6)), dim=1)
        return torch.exp(-root_ori_error/self.cfg.rewards.tracking_sigma), root_ori_error
    
    def _reward_root_pos(self):
        # Tracking of linear velocity commands (xy axes)
        target_root_pos = self.target_global_pos[:,:3]+ self.env_origins
        root_pos_error = torch.sum(torch.square(target_root_pos - self.root_pos), dim=1)
        return torch.exp(-root_pos_error/self.cfg.rewards.tracking_sigma), root_pos_error
    
    def _reward_global_pos_phc(self):
        # global_body_pos = (self.body_pos_raw - self.env_origins[:,None]).view(-1,60)
        # diff_global_body_pos = torch.mean(torch.clip(torch.square(self.target_global_pos - global_body_pos)-0.000, min=0), dim=1)
        # return torch.exp(-100 * diff_global_body_pos), diff_global_body_pos
        global_body_pos = (self.body_pos_raw - self.env_origins[:,None])
        global_body_pos[...,2] = global_body_pos[...,2] - global_body_pos[...,0:1,2]
        global_body_pos_znorm=global_body_pos.view(-1,60)
        target_global_pos_znorm=self.target_global_pos.view(-1,20,3)
        target_global_pos_znorm[...,2] = target_global_pos_znorm[...,2] - target_global_pos_znorm[...,0:1,2]
        target_global_pos_znorm=target_global_pos_znorm.view(-1,60)
        diff_global_body_pos = torch.mean(torch.clip(torch.square(target_global_pos_znorm - global_body_pos_znorm)-0.0005, min=0), dim=1)
        return torch.exp(-100 * diff_global_body_pos), diff_global_body_pos
    
    def _reward_global_ori_phc(self):
        diff_global_body_rot = broadcast_quat_multiply(vec6d_to_quat(self.target_global_ori.reshape(-1,3,2)).reshape(-1,20,4), 
            diff_rot.quat_conjugate(self.body_ori_raw))
        diff_global_body_angle = diff_rot.quat_to_angle_axis(diff_global_body_rot)[0]
        diff_global_body_angle = torch.mean(torch.clip(torch.square(diff_global_body_angle)-0.005, min=0), dim=1)
        return torch.exp(-10 * diff_global_body_angle), diff_global_body_angle
    
    def _reward_global_vel_phc(self):
        diff_global_body_vel = torch.mean(torch.square(self.target_global_vel - self.body_vel_raw.reshape(-1,60)), dim=1)
        return torch.exp(-0.1 * diff_global_body_vel), diff_global_body_vel
    
    def _reward_global_ang_vel_phc(self):
        diff_global_body_ang_vel = torch.mean(torch.square(self.target_global_ang_vel - self.body_ang_vel_raw.reshape(-1,60)), dim=1)
        return torch.exp(-0.1 * diff_global_body_ang_vel), diff_global_body_ang_vel
    
    def _reward_power_phc(self):
        # Penalize energy cost
        power = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        power[self.episode_length_buf <= 5] = 0
        return power, power

    def _reward_local_pos_phc(self):
        local_body_pos = self.body_pos.view(-1,3 * (self.num_dofs + 1))
        target_local_body_pos = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                     (self.target_global_pos.reshape(-1,self.num_dofs + 1,3) - 
                                                      self.target_global_pos[:,None,:3])).reshape(-1, 3 * (self.num_dofs + 1))
        diff_local_body_pos = torch.mean(torch.square(target_local_body_pos - local_body_pos), dim=1)
        return torch.exp(-100 * diff_local_body_pos), diff_local_body_pos
    
    def _reward_local_ori_phc(self):
        local_body_ori = broadcast_quat_multiply(self.heading_quat_inv, 
                                                 self.body_ori_raw)
        target_local_body_ori = broadcast_quat_multiply(self.target_heading_quat_inv, 
                                                        vec6d_to_quat(self.target_global_ori.reshape(-1,self.num_dofs + 1,3,2)).reshape(-1,20,4))
        diff_local_body_rot = broadcast_quat_multiply(target_local_body_ori, 
            diff_rot.quat_conjugate(local_body_ori))
        diff_local_body_angle = diff_rot.quat_to_angle_axis(diff_local_body_rot)[0]
        diff_local_body_angle = torch.mean(torch.square(diff_local_body_angle), dim=1)
        return torch.exp(-10 * diff_local_body_angle), diff_local_body_angle
    
    def _reward_local_vel_phc(self):
        target_local_vel = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                self.target_global_vel.reshape(-1,self.num_dofs + 1,3)).reshape(-1, 3 * (self.num_dofs + 1))
        diff_local_body_vel = torch.mean(torch.square(target_local_vel - self.body_vel.reshape(-1,60)), dim=1)
        return torch.exp(-0.1 * diff_local_body_vel), diff_local_body_vel
    
    def _reward_local_ang_vel_phc(self):
        target_local_ang_vel = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                self.target_global_ang_vel.reshape(-1,self.num_dofs + 1,3)).reshape(-1, 3 * (self.num_dofs + 1))
        diff_local_body_ang_vel = torch.mean(torch.square(target_local_ang_vel - self.body_ang_vel.reshape(-1,60)), dim=1)
        return torch.exp(-0.1 * diff_local_body_ang_vel), diff_local_body_ang_vel
    
    