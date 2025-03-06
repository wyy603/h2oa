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
from legged_gym.privileged.observation import HistoryBuffer

EPS=1e-4

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
        self.play = hasattr(self.cfg,'play') and self.cfg.play
        self.test = hasattr(self.cfg,'test') and self.cfg.test
        self.debug_viz = (self.play) or (self.test)
        if self.test:
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
        self._init_target_pkls()

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
        # ############### state ################
        # if self.cfg.env.obs.dof_pos:
        #     self.num_obs += self.cfg.env.num_dofs
        # if self.cfg.env.obs.dof_vel:
        #     self.num_obs += self.cfg.env.num_dofs

        # if self.cfg.env.obs.body_pos:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.body_ori:
        #     self.num_obs += 6*self.cfg.env.num_bodies
        # if self.cfg.env.obs.body_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.body_ang_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies

        # if self.cfg.env.obs.root_pos:
        #     self.num_obs += 3
        # if self.cfg.env.obs.root_ori:
        #     self.num_obs += 6
        # if self.cfg.env.obs.root_vel:
        #     self.num_obs += 3
        # if self.cfg.env.obs.root_ang_vel:
        #     self.num_obs += 3
        # if self.cfg.env.obs.root_high:
        #     self.num_obs += 1

        # if self.cfg.env.obs.last_action:
        #     self.num_obs += self.cfg.env.num_dofs

        # if self.cfg.env.obs.base_orn_rp:
        #     self.num_obs += 2
        # if self.cfg.env.obs.base_lin_vel:
        #     self.num_obs += 3
        # if self.cfg.env.obs.base_ang_vel:
        #     self.num_obs += 3
        # if self.cfg.env.obs.projected_gravity:
        #     self.num_obs += 3
        # if self.cfg.env.obs.commands:
        #     self.num_obs += 3

        # ########### target ###########

        # if self.cfg.env.obs.target_dof_pos:
        #     self.num_obs += self.cfg.env.num_dofs
        # if self.cfg.env.obs.target_dof_vel:
        #     self.num_obs += self.cfg.env.num_dofs

        # if self.cfg.env.obs.target_body_pos:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.target_body_ori:
        #     self.num_obs += 6*self.cfg.env.num_bodies
        # if self.cfg.env.obs.target_body_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.target_body_ang_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        
        # if self.cfg.env.obs.target_global_pos:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.target_global_ori:
        #     self.num_obs += 6*self.cfg.env.num_bodies
        # if self.cfg.env.obs.target_global_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.target_global_ang_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        
        # if self.cfg.env.obs.diff_local_pos:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.diff_local_ori:
        #     self.num_obs += 6*self.cfg.env.num_bodies
        # if self.cfg.env.obs.diff_local_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.diff_local_ang_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        
        # if self.cfg.env.obs.diff_global_pos:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.diff_global_ori:
        #     self.num_obs += 6*self.cfg.env.num_bodies
        # if self.cfg.env.obs.diff_global_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        # if self.cfg.env.obs.diff_global_ang_vel:
        #     self.num_obs += 3*self.cfg.env.num_bodies
        
        # if self.cfg.env.obs.target_root_pos:
        #     self.num_obs += 3
        # if self.cfg.env.obs.target_root_ori:
        #     self.num_obs += 6
        # if self.cfg.env.obs.target_root_vel:
        #     self.num_obs += 3
        # if self.cfg.env.obs.target_root_ang_vel:
        #     self.num_obs += 3
        if self.cfg.env.obs.dof_pos:
            self.num_obs += 19*2
        elif self.cfg.env.obs.body_pos:
            self.num_obs += 22*3 + 22*6
        self.num_obs += 19+3+3
        if self.cfg.env.obs.target_dof_pos:
            self.num_obs += 19*4
        elif self.cfg.env.obs.target_body_pos:
            self.num_obs += 22*3*2 + 22*6*2
        self.num_obs += 6+3+3

        # self.num_obs += 33*22+4
    


        self.num_privileged_obs = self.num_obs + 33*22+4 + cfg.env.num_privileged_obs if cfg.env.num_privileged_obs is not None else None
        self.num_actions = cfg.env.num_actions
        self.obs_context_len = cfg.env.obs_context_len
        print('self.num_obs: ', self.num_obs)
        print('self.num_privileged_obs: ', self.num_privileged_obs)

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
        
        self.scaling_factor = 1 if self.test else 0.1

    def get_observations(self):  
        if self.cfg.env.policy_name == "ActorCritic":
            return self.obs_buf
        else:
            return self.obs_history_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf
        # return None

    def _init_target_pkls(self):
        self.target_jt_seq, self.target_global_seq, self.target_seq_len, self.motion_start_id = load_target_pkl_concat(self.device, self.cfg.human.filename)
        self.num_target_seq = self.target_seq_len.shape[0]
        print(f"Loaded target joint trajectories of shape {self.target_seq_len.shape}")
        # assert(self.dim_target_jt == self.num_dofs)
        if self.test:
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
            self.extras['obs_global_buf'] = torch.cat([obs_global_pos, obs_global_ori, obs_global_vel, obs_global_ang_vel], dim=1) # NOTE(xh) IMPORTANT BUG!!!!!!!!!!!! reshape first
            self.extras['obs_jt_buf'] = torch.cat([self.dof_pos,self.dof_vel], dim=1).detach().cpu().clone()

        # if self.cfg.human.resample_on_env_reset:
        self.extras['succ_id'] = self.target_jt_i[jt_eps_end_bool]
        if self.test:
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
            self.target_global[:,:3*self.num_bodies], self.target_global[:,3*self.num_bodies:9*self.num_bodies], self.target_global[:,9*self.num_bodies:12*self.num_bodies], self.target_global[:,12*self.num_bodies:15*self.num_bodies]
        self.target_heading_quat_inv = diff_rot.calc_heading_quat_inv(vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)))[:,None]


        t2 = time.time()
        delayed_target_jt_j_cpu = torch.maximum(target_jt_j_cpu - self.delayed_obs_target_jt_steps_int, torch.tensor(0))

        # self.delayed_obs_target_jt = torch.stack([self.target_jt_seq[i][j] for i, j in zip(target_jt_i_cpu, delayed_target_jt_j_cpu)], dim=0).to(self.device)
        self.delayed_obs_target_jt = self.target_jt_seq[self.motion_start_id[target_jt_i_cpu]+delayed_target_jt_j_cpu]#.to(self.device)
        self.delayed_obs_target_jt_pos, self.delayed_obs_target_jt_vel = self.delayed_obs_target_jt[:,:19], self.delayed_obs_target_jt[:,19:]
   
        # self.delayed_obs_target_global = torch.stack([self.target_global_seq[i][j] for i, j in zip(target_jt_i_cpu, delayed_target_jt_j_cpu)], dim=0).to(self.device)
        self.delayed_obs_target_global = self.target_global_seq[self.motion_start_id[target_jt_i_cpu]+delayed_target_jt_j_cpu]#.to(self.device)
        self.delayed_obs_target_global_pos, self.delayed_obs_target_global_ori, self.delayed_obs_target_global_vel, self.delayed_obs_target_global_ang_vel =\
            self.delayed_obs_target_global[:,:3*self.num_bodies], self.delayed_obs_target_global[:,3*self.num_bodies:9*self.num_bodies], self.delayed_obs_target_global[:,9*self.num_bodies:12*self.num_bodies], self.delayed_obs_target_global[:,12*self.num_bodies:15*self.num_bodies]
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
            if self.cfg.domain_rand.randomize_proprio_latency:
                self.update_proprio_latency_buf()
        
        # breakpoint()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

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
        self.gym.refresh_force_sensor_tensor(self.sim)

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
        self.body_pos = broadcast_quat_apply(self.heading_quat_inv, self.body_pos_raw - self.root_pos[:,None]).reshape(-1, 3 * self.num_bodies)
        self.body_ori_quat = flip_quat_by_w(broadcast_quat_multiply(self.heading_quat_inv, self.body_ori_raw))
        self.body_ori = quat_to_vec6d(self.body_ori_quat).reshape(-1, 6 * self.num_bodies)
        self.body_vel = broadcast_quat_apply(self.heading_quat_inv, self.body_vel_raw).reshape(-1, 3 * self.num_bodies)
        self.body_ang_vel = broadcast_quat_apply(self.heading_quat_inv, self.body_ang_vel_raw).reshape(-1, 3 * self.num_bodies)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination() # self.reset_buf = termination_contact_buf | r_threshold_buff | p_threshold_buff | z_threshold_buff | self.time_out_buf
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.update_target_pkls(env_ids)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids) # Will body info updated here?
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.heading_quat[:] = diff_rot.calc_heading_quat(self.base_quat)[:,None]
        self.heading_quat_inv[:] = diff_rot.calc_heading_quat_inv(self.base_quat)[:,None]
        self.body_pos = broadcast_quat_apply(self.heading_quat_inv, self.body_pos_raw - self.root_pos[:,None]).reshape(-1, 3 * self.num_bodies)
        self.body_ori_quat = flip_quat_by_w(broadcast_quat_multiply(self.heading_quat_inv, self.body_ori_raw))
        self.body_ori = quat_to_vec6d(self.body_ori_quat).reshape(-1, 6 * self.num_bodies)
        self.body_vel = broadcast_quat_apply(self.heading_quat_inv, self.body_vel_raw).reshape(-1, 3 * self.num_bodies)
        self.body_ang_vel = broadcast_quat_apply(self.heading_quat_inv, self.body_ang_vel_raw).reshape(-1, 3 * self.num_bodies)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:].clone()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        

    def check_termination(self):
        """ Check if environments need to be reset
        """

        if (self.test) or (self.play):
            contact_threshold = 9999999999999
            r_threshold = 9999999999999
            p_threshold = 9999999999999
            z_threshold = 0
        else:
            contact_threshold = self.cfg.termination.contact_threshold
            r_threshold = self.cfg.termination.r_threshold * self.scaling_factor * 10
            p_threshold = self.cfg.termination.p_threshold * self.scaling_factor * 10
            z_threshold = self.cfg.termination.z_threshold / (self.scaling_factor * 10)
            # termination_threshold = self.cfg.termination.termination_distance / self.scaling_factor

        termination_threshold = self.cfg.termination.termination_distance
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > contact_threshold, dim=1)

        r, p = self.base_orn_rp[:, 0], self.base_orn_rp[:, 1]
        z = self.root_states[:, 2]


        r_threshold_buff = r.abs() > r_threshold
        p_threshold_buff = p.abs() > p_threshold
        z_threshold_buff = z < z_threshold
        has_fallen = torch.zeros_like(r_threshold_buff)

        # power = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        # power[self.episode_length_buf <= 5] = 0
        # termination_power_buf = power > 4000

        self.reset_buf = termination_contact_buf | r_threshold_buff | p_threshold_buff | z_threshold_buff #| termination_power_buf

        if hasattr(self.cfg.termination,'enable_early_termination') and self.cfg.termination.enable_early_termination:

            if self.cfg.env.obs.target_global_pos or True:
                global_body_pos = (self.body_pos_raw - self.env_origins[:,None])
                target_global_pos=self.target_global_pos.view(-1,self.num_bodies,3)
                if self.cfg.env.z_norm:
                    global_body_pos[...,2] = global_body_pos[...,2] - global_body_pos[...,0:1,2]
                    target_global_pos[...,2] = target_global_pos[...,2] - target_global_pos[...,0:1,2]
                # global_body_pos_znorm=global_body_pos
                # target_global_pos_znorm=target_global_pos
                # global_body_pos = (self.body_pos_raw - self.env_origins[:,None])
                # mean_global_distance = torch.mean(torch.norm(global_body_pos - self.target_global_pos.view(-1,self.num_bodies,3), dim=-1), dim=-1)
                # breakpoint()
                mean_global_distance = torch.mean(torch.norm(global_body_pos - target_global_pos, dim=-1), dim=-1)
                if self.debug_viz:
                    has_fallen = torch.any(torch.norm(global_body_pos - target_global_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_threshold, dim=-1)  # using mean
                else:
                    has_fallen = torch.any(torch.norm(global_body_pos - target_global_pos, dim=-1) > termination_threshold, dim=-1)  # using max
                self.extras['mean_global_distance'] = mean_global_distance
            elif self.cfg.env.obs.target_body_pos or self.cfg.env.obs.target_dof_pos:
                local_body_pos = self.body_pos.reshape(-1,self.num_bodies,3)
                target_local_body_pos = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                            (self.target_global_pos.reshape(-1,self.num_bodies,3) - 
                                                            self.target_global_pos[:,None,:3]))
                if self.debug_viz:
                    has_fallen = torch.any(torch.norm(local_body_pos - target_local_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_threshold, dim=-1)  # using mean
                else:
                    has_fallen = torch.any(torch.norm(local_body_pos - target_local_body_pos, dim=-1) > termination_threshold, dim=-1)  # using max
            else:
                raise NotImplementedError
            # target_root_pos = self.target_global_pos[:,:3]+ self.env_origins
            # root_pos_error = torch.norm(target_root_pos - self.root_pos, dim=1)
            # has_fallen = has_fallen | (root_pos_error > termination_threshold)
            has_fallen = has_fallen & (self.episode_length_buf > self.cfg.env.extend_frames) 
            self.reset_buf = self.reset_buf | has_fallen #| self.time_out_buf #check in update_target_jt
             #| self.time_out_buf
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
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        if not self.debug_viz:
            if self.episode_length_buf.float().mean().item() < self.cfg.env.curriculum_lowerband:
                self.scaling_factor = max(self.scaling_factor * (1-EPS), .1)
            elif self.episode_length_buf.float().mean().item()  > self.cfg.env.curriculum_upperband:
                self.scaling_factor = min(self.scaling_factor * (1+EPS), 1.)
        self.extras['episode_length_current'] = self.episode_length_buf.float().mean()
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids])
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
                unscaled_rew *= (self.scaling_factor - 0.09)
            rew = unscaled_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metrics[name] = metric.to(self.rew_buf).mean().item()
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def update_proprio_latency_buf(self): # TODO check if change real-time
        self.heading_quat[:] = diff_rot.calc_heading_quat(self.base_quat)[:,None]
        self.heading_quat_inv[:] = diff_rot.calc_heading_quat_inv(self.base_quat)[:,None]
        self.body_pos = broadcast_quat_apply(self.heading_quat_inv, self.body_pos_raw - self.root_pos[:,None]).reshape(-1, 3 * self.num_bodies)
        self.body_ori_quat = flip_quat_by_w(broadcast_quat_multiply(self.heading_quat_inv, self.body_ori_raw))
        self.body_ori = quat_to_vec6d(self.body_ori_quat).reshape(-1, 6 * self.num_bodies)
        self.base_ang_vel = quat_rotate_inverse(self.heading_quat[:,0], self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.dof_pos_buf.update(self.dof_pos)
        self.dof_vel_buf.update(self.dof_vel)
        self.body_pos_buf.update(self.body_pos)
        self.body_ori_buf.update(self.body_ori)
        self.base_ang_vel_buf.update(self.base_ang_vel)
        self.projected_gravity_buf.update(self.projected_gravity)

    def compute_observations(self):
        """ Computes observations
        """
            
        B, _ = self.dof_pos.shape
        
        obs_list = []
        pri_obs_list = []
        # TODO(xh) body: root frame, local: heading frame, global: world frame
        if self.cfg.domain_rand.randomize_proprio_latency:
            # pass
            simsteps = self.proprio_latency_simsteps.unsqueeze(1).unsqueeze(2)
            self.dof_pos_latency = torch.gather(self.dof_pos_buf.get_history_buffer(), 1, simsteps.expand(-1, -1, self.num_dofs)).squeeze(1)
            self.dof_vel_latency = torch.gather(self.dof_vel_buf.get_history_buffer(), 1, simsteps.expand(-1, -1, self.num_dofs)).squeeze(1)
            self.body_pos_latency = torch.gather(self.body_pos_buf.get_history_buffer(), 1, simsteps.expand(-1, -1, 3 * self.num_bodies)).squeeze(1)
            self.body_ori_latency = torch.gather(self.body_ori_buf.get_history_buffer(), 1, simsteps.expand(-1, -1, 6 * self.num_bodies)).squeeze(1)
            self.base_ang_vel_latency = torch.gather(self.base_ang_vel_buf.get_history_buffer(), 1, simsteps.expand(-1, -1, 3)).squeeze(1)
            self.projected_gravity_latency = torch.gather(self.projected_gravity_buf.get_history_buffer(), 1, simsteps.expand(-1, -1, 3)).squeeze(1)
        else:
            self.dof_pos_latency = self.dof_pos
            self.dof_vel_latency = self.dof_vel
            self.body_pos_latency = self.body_pos
            self.body_ori_latency = self.body_ori
            # self.body_ori_quat_latency = self.body_ori_quat
            self.base_ang_vel_latency = self.base_ang_vel
            self.projected_gravity_latency = self.projected_gravity
        ############### state ################

        # if self.cfg.env.obs.dof_pos: # num_dofs local
        #     obs_list.append(self.dof_pos * self.obs_scales.dof_pos)
        # if self.cfg.env.obs.dof_vel: # num_dofs
        #     obs_list.append(self.dof_vel * self.obs_scales.dof_vel)

    # if self.cfg.env.obs.body_pos: # 3 * (num_dofs + 1) wtf heading frame
        pri_obs_list.append(self.body_pos * self.obs_scales.body_pos)
    # if self.cfg.env.obs.body_ori: # 6 * (num_dofs + 1) wtf heading frame
        pri_obs_list.append(self.body_ori * self.obs_scales.body_ori)
    # if self.cfg.env.obs.body_vel: # 3 * (num_dofs + 1) wtf heading frame
        pri_obs_list.append(self.body_vel * self.obs_scales.body_vel)
    # if self.cfg.env.obs.body_ang_vel: # 3 * (num_dofs + 1) wtf heading frame
        pri_obs_list.append(self.body_ang_vel * self.obs_scales.body_ang_vel) 

        # if self.cfg.env.obs.root_pos: # 3 global w/o env origins
        #     obs_list.append((self.root_pos-self.env_origins) * self.obs_scales.root_pos)
        # if self.cfg.env.obs.root_ori: # 6 global
        #     obs_list.append(quat_to_vec6d(self.root_ori).reshape([-1,6]) * self.obs_scales.root_ori)
        # if self.cfg.env.obs.root_vel: # 3 global
        #     obs_list.append(self.root_vel * self.obs_scales.root_vel)
        # if self.cfg.env.obs.root_ang_vel: # 3 global
        #     obs_list.append(self.root_ang_vel * self.obs_scales.root_ang_vel)  


        # if self.cfg.env.obs.root_high: # 1
        pri_obs_list.append((self.root_pos-self.env_origins)[...,-1:] * self.obs_scales.root_pos)

        # if self.cfg.env.obs.last_action: # num_dofs
        #     obs_list.append(self.actions * self.obs_scales.last_action)

        # if self.cfg.env.obs.base_orn_rp: # 2
        #     obs_list.append(self.base_orn_rp * self.obs_scales.base_orn_rp)
        # if self.cfg.env.obs.base_lin_vel: # 3 wtf heading frame
        pri_obs_list.append(self.base_lin_vel * self.obs_scales.base_lin_vel) 
        # if self.cfg.env.obs.base_ang_vel: # 3 wtf heading frame
        #     obs_list.append(self.base_ang_vel * self.obs_scales.base_ang_vel) 
        # if self.cfg.env.obs.projected_gravity: # wtf root frame
        #     obs_list.append(self.projected_gravity * self.obs_scales.base_ang_vel)

        # if self.cfg.env.obs.commands: # 3
        #     obs_list.append(self.commands[:, :3] * self.commands_scale[:3])  
        if self.cfg.env.obs.dof_pos:
            # latency: self.dof_pos self.dof_vel
            obs_list.append(self.dof_pos_latency * self.obs_scales.dof_pos) # 19
            obs_list.append(self.dof_vel_latency * self.obs_scales.dof_vel) # 19
        elif self.cfg.env.obs.body_pos:
            # latency: self.body_pos self.body_ori
            obs_list.append(self.body_pos_latency * self.obs_scales.body_pos) # 22*3
            obs_list.append(self.body_ori_latency * self.obs_scales.body_ori) # 22*6
        else:
            raise Exception("Invalid observation configuration")

        # latency: self.base_ang_vel self.projected_gravity
        obs_list.append(self.actions * self.obs_scales.last_action) # 19
        obs_list.append(self.base_ang_vel_latency * self.obs_scales.base_ang_vel) # 3
        obs_list.append(self.projected_gravity_latency * self.obs_scales.base_ang_vel) # 3

        ##################### task #####################

        # if self.cfg.env.obs.target_dof_pos: # num_dofs
        #     obs_list.append(self.delayed_obs_target_jt_pos * self.obs_scales.target_dof_pos)
        # if self.cfg.env.obs.target_dof_vel: # num_dofs
        #     obs_list.append(self.delayed_obs_target_jt_vel * self.obs_scales.target_dof_vel)

        # if self.cfg.env.obs.target_body_pos: # 3 * (num_dofs + 1) wtf heading frame
        #     target_body_pos = self.delayed_obs_target_global_pos.reshape(-1, self.num_bodies, 3)  - self.delayed_obs_target_global_pos[:,None,:3]
        #     target_body_pos = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv, target_body_pos).reshape(-1, 3 * (self.num_bodies))
        #     obs_list.append(target_body_pos * self.obs_scales.target_body_pos)
        # if self.cfg.env.obs.target_body_ori: # 6 * (num_dofs + 1) wtf heading frame
        #     target_body_ori = broadcast_quat_multiply(self.delayed_obs_target_heading_quat_inv,
        #         vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(B,self.num_bodies,3,2)))
        #     target_body_ori = quat_to_vec6d(target_body_ori).reshape(-1, 6 * (self.num_bodies))
        #     obs_list.append(target_body_ori * self.obs_scales.target_body_ori)
        # if self.cfg.env.obs.target_body_vel: # 3 * (num_dofs + 1) none
        #     obs_list.append(self.delayed_obs_target_body_vel * self.obs_scales.target_body_vel)
        # if self.cfg.env.obs.target_body_ang_vel: # 3 * (num_dofs + 1) none
        #     obs_list.append(self.delayed_obs_target_body_ang_vel * self.obs_scales.target_body_ang_vel) 

        # if self.cfg.env.obs.target_global_pos: # 3 * (num_dofs + 1) wtf heading frame
        target_global_pos = self.delayed_obs_target_global_pos.reshape(-1, self.num_bodies, 3) - (self.root_pos-self.env_origins)[:,None]
        target_global_pos = broadcast_quat_apply(self.heading_quat_inv, target_global_pos).reshape(-1, 3 * (self.num_bodies))
        pri_obs_list.append(target_global_pos * self.obs_scales.target_global_pos)
        # if self.cfg.env.obs.target_global_ori: # 6 * (num_dofs + 1) wtf heading frame
        target_global_ori = broadcast_quat_multiply(self.heading_quat_inv,
            vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(B,self.num_bodies,3,2)))
        target_global_ori = quat_to_vec6d(target_global_ori).reshape(-1, 6 * (self.num_bodies))
        pri_obs_list.append(target_global_ori * self.obs_scales.target_global_ori)
        # if self.cfg.env.obs.target_global_vel: # 3 * (num_dofs + 1)
        #     obs_list.append(self.delayed_obs_target_global_vel * self.obs_scales.target_global_vel)
        # if self.cfg.env.obs.target_global_ang_vel: # 3 * (num_dofs + 1)
        #     obs_list.append(self.delayed_obs_target_global_ang_vel * self.obs_scales.target_global_ang_vel) 



        # if self.cfg.env.obs.diff_local_pos: # 3 * (num_dofs + 1)
        #     target_body_pos = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv,
        #         (self.delayed_obs_target_global_pos.reshape(-1, self.num_bodies, 3) - self.delayed_obs_target_global_pos[:,None,:3]))
        #     diff_local_pos = (target_body_pos.reshape(-1, 3 * (self.num_bodies)) - self.body_pos)
        #     obs_list.append(diff_local_pos * self.obs_scales.diff_local_pos)
        # if self.cfg.env.obs.diff_local_ori: # 6 * (num_dofs + 1) 
        #     local_body_ori = broadcast_quat_multiply(self.heading_quat_inv, 
        #                                          self.body_ori_raw)
        #     target_local_body_ori = broadcast_quat_multiply(self.delayed_obs_target_heading_quat_inv, 
        #                                                     vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(-1,self.num_bodies,3,2)).reshape(-1,self.num_bodies,4))
        #     diff_local_body_rot = broadcast_quat_multiply(target_local_body_ori, 
        #         diff_rot.quat_conjugate(local_body_ori))
        #     diff_local_body_rot = quat_to_vec6d(diff_local_body_rot).reshape(B, 6 * (self.num_bodies))
        #     obs_list.append(diff_local_body_rot * self.obs_scales.diff_local_ori)
        # if self.cfg.env.obs.diff_local_vel: # 3 * (num_dofs + 1)
        #     target_local_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv, 
        #                                         self.delayed_obs_target_global_vel.reshape(-1,self.num_bodies,3)).reshape(-1, 3 * (self.num_bodies))
        #     diff_local_vel = target_local_vel - self.body_vel
        #     obs_list.append(diff_local_vel * self.obs_scales.diff_local_vel)
        # if self.cfg.env.obs.diff_local_ang_vel: # 3 * (num_dofs + 1)
        #     target_local_ang_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv, 
        #                                         self.delayed_obs_target_global_ang_vel.reshape(-1,self.num_bodies,3)).reshape(-1, 3 * (self.num_bodies))
        #     diff_local_ang_vel = target_local_ang_vel - self.body_ang_vel
        #     obs_list.append(diff_local_ang_vel * self.obs_scales.diff_local_ang_vel)

        # if self.cfg.env.obs.diff_global_pos: # 3 * (num_dofs + 1)
        diff_global_pos = self.delayed_obs_target_global_pos.reshape(B, self.num_bodies, 3) - (self.body_pos_raw - self.env_origins[:,None])
        diff_global_pos = broadcast_quat_apply(self.heading_quat_inv, diff_global_pos).reshape(-1, 3 * (self.num_bodies))
        pri_obs_list.append(diff_global_pos * self.obs_scales.diff_global_pos)
        # if self.cfg.env.obs.diff_global_ori: # 6 * (num_dofs + 1) 
        diff_global_body_rot = broadcast_quat_multiply(vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(B,self.num_bodies,3,2)), 
                                diff_rot.quat_conjugate(self.body_ori_raw))
        diff_global_body_rot = broadcast_quat_multiply(self.heading_quat_inv,
                                broadcast_quat_multiply(diff_global_body_rot, 
                                self.heading_quat))
        diff_global_body_rot = quat_to_vec6d(diff_global_body_rot).reshape(B, 6 * (self.num_bodies))
        pri_obs_list.append(diff_global_body_rot * self.obs_scales.diff_global_ori)
        # if self.cfg.env.obs.diff_global_vel: # 3 * (num_dofs + 1)
        #     diff_global_vel = self.delayed_obs_target_global_vel.reshape(B, self.num_bodies, 3) - self.body_vel_raw
        #     diff_global_vel = broadcast_quat_apply(self.heading_quat_inv, diff_global_vel).reshape(-1, 3 * (self.num_bodies))
        #     obs_list.append(diff_global_vel * self.obs_scales.diff_global_vel)
        # if self.cfg.env.obs.diff_global_ang_vel: # 3 * (num_dofs + 1)
        #     diff_global_ang_vel = self.delayed_obs_target_global_ang_vel.reshape(B, self.num_bodies, 3) - self.body_ang_vel_raw
        #     diff_global_ang_vel = broadcast_quat_apply(self.heading_quat_inv, diff_global_ang_vel).reshape(-1, 3 * (self.num_bodies))
        #     obs_list.append(diff_global_ang_vel * self.obs_scales.diff_global_ang_vel)


        # if self.cfg.env.obs.target_root_pos: # 3 global
        #     obs_list.append(self.delayed_obs_target_global_pos[:,:3] * self.obs_scales.target_root_pos)
        # if self.cfg.env.obs.target_root_ori: # 6 global
        #     obs_list.append(self.delayed_obs_target_global_ori[:,:6] * self.obs_scales.target_root_ori)
        # if self.cfg.env.obs.target_root_vel: # 3 local
        #     target_root_local_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv.squeeze(1), 
        #                                         self.delayed_obs_target_global_vel[:,:3])
        #     diff_root_local_vel = target_root_local_vel - self.base_lin_vel
        #     obs_list.append(diff_root_local_vel * self.obs_scales.target_root_vel)
        #     # obs_list.append(self.delayed_obs_target_body_vel[:,:3] * self.obs_scales.target_root_vel)
        # if self.cfg.env.obs.target_root_ang_vel: # 3 local
        #     target_root_local_ang_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv.squeeze(1), 
        #                                         self.delayed_obs_target_global_ang_vel[:,:3])
        #     diff_root_local_ang_vel = target_root_local_ang_vel - self.base_ang_vel
        #     obs_list.append(diff_root_local_ang_vel * self.obs_scales.target_root_ang_vel) 
        #     # obs_list.append(self.delayed_obs_target_body_ang_vel[:,:3] * self.obs_scales.target_root_ang_vel) 
        if self.cfg.env.obs.target_dof_pos:
            # latency: self.dof_pos self.dof_vel
            obs_list.append(self.delayed_obs_target_jt_pos * self.obs_scales.target_dof_pos)
            obs_list.append(self.delayed_obs_target_jt_vel * self.obs_scales.target_dof_vel)

            obs_list.append((self.delayed_obs_target_jt_pos-self.dof_pos_latency) * self.obs_scales.target_dof_pos)
            obs_list.append((self.delayed_obs_target_jt_vel-self.dof_vel_latency) * self.obs_scales.target_dof_vel)

        elif self.cfg.env.obs.target_body_pos:
            # latency: self.body_pos self.body_ori
            target_body_pos = self.delayed_obs_target_global_pos.reshape(-1, self.num_bodies, 3)  - self.delayed_obs_target_global_pos[:,None,:3]
            target_body_pos = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv, target_body_pos).reshape(-1, 3 * (self.num_bodies))
            obs_list.append(target_body_pos * self.obs_scales.target_body_pos) # 22x3
            target_body_ori_quat = broadcast_quat_multiply(self.delayed_obs_target_heading_quat_inv,
                vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(B,self.num_bodies,3,2)))
            target_body_ori = quat_to_vec6d(target_body_ori_quat).reshape(-1, 6 * (self.num_bodies))
            obs_list.append(target_body_ori * self.obs_scales.target_body_ori) # 22x6

            diff_local_pos = (target_body_pos - self.body_pos_latency)
            obs_list.append(diff_local_pos * self.obs_scales.diff_local_pos) # 22x3
            body_ori_quat = vec6d_to_quat(self.body_ori_latency.reshape(B,self.num_bodies,3,2))
            diff_local_body_rot = broadcast_quat_multiply(target_body_ori_quat, 
                diff_rot.quat_conjugate(body_ori_quat))
            diff_local_body_rot = quat_to_vec6d(diff_local_body_rot).reshape(B, 6 * (self.num_bodies))
            obs_list.append(diff_local_body_rot * self.obs_scales.diff_local_ori) # 22x6
        else:
            raise Exception("Invalid observation configuration")

        target_root_ori_quat = broadcast_quat_multiply(self.delayed_obs_target_heading_quat_inv,
            vec6d_to_quat(self.delayed_obs_target_global_ori.reshape(B,self.num_bodies,3,2)))[:,0]
        target_root_ori=quat_to_vec6d(target_root_ori_quat).reshape(-1, 6)
        obs_list.append(target_root_ori * self.obs_scales.target_root_ori) # 6

        target_root_local_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv.squeeze(1), 
                                            self.delayed_obs_target_global_vel[:,:3])
        obs_list.append(target_root_local_vel * self.obs_scales.target_root_vel) # 3
        target_root_local_ang_vel = broadcast_quat_apply(self.delayed_obs_target_heading_quat_inv.squeeze(1), 
                                            self.delayed_obs_target_global_ang_vel[:,:3])
        obs_list.append(target_root_local_ang_vel * self.obs_scales.target_root_ang_vel) # 3

        self.obs_buf = torch.cat(obs_list, dim=-1)
        # pri_obs_buf = torch.cat(pri_obs_list, dim=-1)

        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.cfg.noise.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
        # self.obs_history_buf = torch.cat([ # TODO(xh)
        #     self.obs_history_buf[:, 1:],
        #     self.obs_buf.unsqueeze(1)
        # ], dim=1)

        if self.privileged_obs_buf is not None:
            body_height = self.root_pos[:, 2].unsqueeze(1)
            foot_height = self.body_pos_raw[:, self.feet_indices, 2].view(self.num_envs, -1)
            foot_velocities = self.body_vel_raw[:, self.feet_indices].view(self.num_envs, -1)
            foot_contact_force = self.force_sensor_tensor[:, :, 2]
            foot_contact_state = self.force_sensor_tensor.norm(dim=-1) > 1.5

            self.privileged_obs_buf = torch.cat((
                # privileged
                *pri_obs_list,
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.obs_buf.clone(),  # proprioceptive info
                self.friction_coeffs.to(self.device).squeeze(-1),  # 1
                self.mass_params_tensor,  # 4
                self.motor_strength - 1,  # 12
                foot_velocities,  # 6
                foot_contact_force / 10,  # 2
                foot_contact_state,  # 2
                body_height,  # 1
                foot_height,  # 2
                # terrain_heights, # only with vision
            ), dim=-1)

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
            self.terrain = Terrain(self.cfg.terrain)
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
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        else:
            if env_id == 0:
                self.friction_coeffs = torch.ones((self.num_envs, 1, 1))

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
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1,))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_base_com:
            rng_com_x = self.cfg.domain_rand.added_com_range_x
            rng_com_y = self.cfg.domain_rand.added_com_range_y
            rng_com_z = self.cfg.domain_rand.added_com_range_z
            rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]],
                                         [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3,))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)

        mass_params = np.hstack([rand_mass, rand_com])
        return props, mass_params
    
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
        if hasattr(self.cfg.env, 'target_init') and self.cfg.env.target_init:# and (np.random.rand() <= self.scaling_factor):
            assert (self.target_jt_j[env_ids] == 0).all(), "target_jt_j should be zero"
            dof_state_copy[..., 0][env_ids] = self.target_jt_pos[env_ids] # NOTE(xh) self.target_jt_pos should update before reset dof
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
        if hasattr(self.cfg.env, 'target_init') and self.cfg.env.target_init:#  and (np.random.rand() <= self.scaling_factor):
            assert self.cfg.env.target_heading_init
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
        # breakpoint()
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim) # (2048, 13)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # 
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.force_sensor_tensor = gymtorch.wrap_tensor(
            force_sensor_tensor).view(self.num_envs, len(self.force_sensor_dict), 6)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.body_state = gymtorch.wrap_tensor(body_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.body_pos_raw = self.body_state.view(self.num_envs, self.num_bodies, -1)[..., :3]
        self.body_ori_raw = self.body_state.view(self.num_envs, self.num_bodies, -1)[..., 3:7]
        self.body_vel_raw = self.body_state.view(self.num_envs, self.num_bodies, -1)[..., 7:10]
        self.body_ang_vel_raw = self.body_state.view(self.num_envs, self.num_bodies, -1)[..., 10:]

        self.base_quat = self.root_states[:, 3:7]
        
        self.root_pos = self.root_states[:, :3]
        self.root_ori = self.root_states[:, 3:7]
        self.root_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        if self.cfg.domain_rand.randomize_proprio_latency:
            low_late_simsteps = int(self.cfg.domain_rand.proprio_latency_range[0] / self.sim_params.dt)
            high_late_simsteps = int(self.cfg.domain_rand.proprio_latency_range[1] / self.sim_params.dt) + 1
            self.proprio_latency_simsteps = torch.randint(
                low=low_late_simsteps, high=high_late_simsteps, size=(self.num_envs,),
                device=self.device, dtype=torch.long)
            self.dof_pos_buf = HistoryBuffer(self.num_envs, high_late_simsteps, self.num_dofs, self.device)
            self.dof_vel_buf = HistoryBuffer(self.num_envs, high_late_simsteps, self.num_dofs, self.device)
            self.body_pos_buf = HistoryBuffer(self.num_envs, high_late_simsteps, self.num_bodies*3, self.device)
            self.body_ori_buf = HistoryBuffer(self.num_envs, high_late_simsteps, self.num_bodies*6, self.device)
            self.base_ang_vel_buf = HistoryBuffer(self.num_envs, high_late_simsteps, 3, self.device)
            self.projected_gravity_buf = HistoryBuffer(self.num_envs, high_late_simsteps, 3, self.device)
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
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
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
        # breakpoint()
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # breakpoint()
        assert(self.num_bodies == len(body_names) and self.num_bodies == self.cfg.env.num_bodies)
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

        self.mass_params_tensor = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)

        self.force_sensor_dict = {}
        for force_sensor_name, force_sensor_config in self.cfg.asset.force_sensor_configs.items():
            link_name = force_sensor_config['link_name']
            position = force_sensor_config['position']
            foot_idx = self.body_names_to_idx[link_name]
            sensor_pose = gymapi.Transform(gymapi.Vec3(*position))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)
            self.force_sensor_dict[force_sensor_name] = sensor_idx
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
            # body_props = self._process_rigid_body_props(body_props, i)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.mass_params_tensor[i] = torch.tensor(mass_params, device=self.device)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch.cat([
                torch_rand_float(self.cfg.domain_rand.leg_motor_strength_range[0],
                                 self.cfg.domain_rand.leg_motor_strength_range[1],
                                 (self.num_envs, self.num_dofs),
                                 device=self.device),
            ], dim=1)
        else:
            self.motor_strength = torch.ones(self.num_envs, self.num_dofs, device=self.device)

       
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False) # TODO(xh)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        def any_in(patterns, string):
            return any(p in string for p in patterns)
        
        self.dof_hip_yaw_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'hip_yaw' in name],
                                                dtype=torch.long, device=self.device, requires_grad=False)
        
        self.dof_hip_roll_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'hip_roll' in name],
                                                dtype=torch.long, device=self.device, requires_grad=False)

        self.dof_ankle_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'ankle' in name],
                                              dtype=torch.long, device=self.device, requires_grad=False)

        self.dof_arm_indices = torch.tensor([i for i, name in enumerate(self.dof_names)
                                             if any_in(['shoulder', 'elbow'], name)],
                                            dtype=torch.long, device=self.device, requires_grad=False)

        self.dof_waist_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'torso' in name],
                                              dtype=torch.long, device=self.device, requires_grad=False)

        
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
            for j in range(self.num_bodies):
                # if j in range(self.num_bodies):
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



    def _reward_alive(self):
        # Alive reward
        # return torch.ones(self.num_envs), torch.ones(self.num_envs)
        bonus = self.episode_length_buf /1000
        return (1. - 1.*self.reset_buf) * (1.+bonus), 1.-1.*self.reset_buf
    
    def _reward_punish_large_vel(self):
        large_x_vel = 1 * (torch.abs(self.base_lin_vel[:, 0]) > 0.8)
        large_y_vel = 1 * (torch.abs(self.base_lin_vel[:, 1]) > 0.8)
        large_z_vel = 1 * (torch.abs(self.base_lin_vel[:, 2]) > 0.8)
        large_roll_vel = 1 * (torch.abs(self.base_ang_vel[:, 0]) > 0.8)
        large_pitch_vel = 1 * (torch.abs(self.base_ang_vel[:, 1]) > 0.8)
        large_yaw_vel = 1 * (torch.abs(self.base_ang_vel[:, 2]) > 0.8)
        exceed_vel_error = large_x_vel + large_y_vel + large_z_vel + large_roll_vel + large_pitch_vel + large_yaw_vel
        return torch.exp(-exceed_vel_error), exceed_vel_error

    def _reward_punish_sliding(self):
        contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        has_feet_contact_forces = 1 * (contact_forces > 50)
        feet_xy_velocities = torch.norm(self.body_vel_raw[:, self.feet_indices, :2], dim=-1)
        has_feet_xy_velocities = 1 * (feet_xy_velocities > 0.1)
        has_sliding = torch.sum(has_feet_contact_forces * has_feet_xy_velocities, dim=-1)
        return torch.exp(-has_sliding), has_sliding
    
    def _reward_contact_forces(self):
        contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        waste_force = contact_forces - self.cfg.rewards.max_contact_force
        rew = torch.sum(torch.clip(waste_force, 0., 500), dim=1)  # exceed 500
        return rew, rew
    
    def _reward_energy(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
        return energy, energy
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        collision = torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
        return collision, collision

    def _reward_action_rate(self):
        # Penalize changes in actions
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate, action_rate

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        out_of_limits = torch.sum(out_of_limits, dim=1)
        return out_of_limits, out_of_limits

    def _reward_action_smoothness(self):
        # ankle
        # diff = torch.square(self.actions[:, self.dof_ankle_indices] - self.last_actions[:, self.dof_ankle_indices])
        # return torch.sum(diff, dim=1)
        vel = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        acc = torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        rew = vel + 0.3 * acc
        return rew, vel
    
    def _reward_arm_waist_ankle_smoothness(self):
        # arm
        arm_err = torch.mean(
            torch.square(self.actions[:, self.dof_arm_indices] - self.last_actions[:, self.dof_arm_indices]), dim=1)

        # waist
        waist_err = torch.mean(
            torch.square(self.actions[:, self.dof_waist_indices] - self.last_actions[:, self.dof_waist_indices]), dim=1)

        # ankle
        ankle_err = torch.mean(
            torch.square(self.actions[:, self.dof_ankle_indices] - self.last_actions[:, self.dof_ankle_indices]), dim=1)

        rew = 15 * arm_err + 5 * waist_err + ankle_err
        return rew, rew
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        dof_vel = torch.sum(torch.square(self.dof_vel), dim=1)
        return dof_vel, dof_vel

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        dof_acc = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        return dof_acc, dof_acc

    def _reward_weighted_torques(self):
        # Penalize torques
        weighted_torques = torch.sum(torch.square(self.torques / self.p_gains.view(1, -1)), dim=1)
        return weighted_torques, torch.mean(self.torques.abs(), dim=-1)



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
        # global_body_pos = (self.body_pos_raw - self.env_origins[:,None]).view(-1,3*self.num_bodies)
        # diff_global_body_pos = torch.mean(torch.clip(torch.square(self.target_global_pos - global_body_pos)-0.000, min=0), dim=1)
        # return torch.exp(-100 * diff_global_body_pos), diff_global_body_pos
        global_body_pos = (self.body_pos_raw - self.env_origins[:,None])
        target_global_pos=self.target_global_pos.view(-1,self.num_bodies,3)
        if self.cfg.env.z_norm:
            global_body_pos[...,2] = global_body_pos[...,2] - global_body_pos[...,0:1,2]
            target_global_pos[...,2] = target_global_pos[...,2] - target_global_pos[...,0:1,2]
        global_body_pos=global_body_pos.view(-1,3*self.num_bodies)
        target_global_pos=target_global_pos.view(-1,3*self.num_bodies)
        diff_global_body_pos = torch.mean(torch.clip(torch.square(target_global_pos - global_body_pos)-0.0001, min=0), dim=1) # .0005
        discrete_pose_reward0 =  0.3*(diff_global_body_pos < 0.002).to(diff_global_body_pos)
        discrete_pose_reward1 =  0.3*(diff_global_body_pos < 0.001).to(diff_global_body_pos)
        return torch.exp(-100 * diff_global_body_pos) + discrete_pose_reward0 + discrete_pose_reward1, diff_global_body_pos
    
    def _reward_global_ori_phc(self):
        diff_global_body_rot = broadcast_quat_multiply(vec6d_to_quat(self.target_global_ori.reshape(-1,3,2)).reshape(-1,self.num_bodies,4), 
            diff_rot.quat_conjugate(self.body_ori_raw))
        diff_global_body_angle = diff_rot.quat_to_angle_axis(diff_global_body_rot)[0]
        diff_global_body_angle = torch.mean(torch.clip(torch.square(diff_global_body_angle)-0.001, min=0), dim=1) # .005
        return torch.exp(-10 * diff_global_body_angle), diff_global_body_angle
    
    def _reward_global_vel_phc(self):
        diff_global_body_vel = torch.mean(torch.square(self.target_global_vel - self.body_vel_raw.reshape(-1,3*self.num_bodies)), dim=1)
        return torch.exp(-0.1 * diff_global_body_vel), diff_global_body_vel
    
    def _reward_global_ang_vel_phc(self):
        diff_global_body_ang_vel = torch.mean(torch.square(self.target_global_ang_vel - self.body_ang_vel_raw.reshape(-1,3*self.num_bodies)), dim=1)
        return torch.exp(-0.1 * diff_global_body_ang_vel), diff_global_body_ang_vel

    def _reward_local_pos_phc(self):
        local_body_pos = self.body_pos.view(-1,3 * (self.num_bodies))
        target_local_body_pos = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                     (self.target_global_pos.reshape(-1,self.num_bodies,3) - 
                                                      self.target_global_pos[:,None,:3])).reshape(-1, 3 * self.num_bodies)
        diff_local_body_pos = torch.mean(torch.square(target_local_body_pos - local_body_pos), dim=1)
        return torch.exp(-100 * diff_local_body_pos), diff_local_body_pos
    
    def _reward_local_ori_phc(self):
        local_body_ori = broadcast_quat_multiply(self.heading_quat_inv, 
                                                 self.body_ori_raw)
        target_local_body_ori = broadcast_quat_multiply(self.target_heading_quat_inv, 
                                                        vec6d_to_quat(self.target_global_ori.reshape(-1,self.num_bodies,3,2)).reshape(-1,self.num_bodies,4))
        diff_local_body_rot = broadcast_quat_multiply(target_local_body_ori, 
            diff_rot.quat_conjugate(local_body_ori))
        diff_local_body_angle = diff_rot.quat_to_angle_axis(diff_local_body_rot)[0]
        diff_local_body_angle = torch.mean(torch.square(diff_local_body_angle), dim=1)
        return torch.exp(-10 * diff_local_body_angle), diff_local_body_angle
    
    def _reward_local_vel_phc(self):
        target_local_vel = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                self.target_global_vel.reshape(-1,self.num_bodies,3)).reshape(-1, 3 * self.num_bodies)
        diff_local_body_vel = torch.mean(torch.square(target_local_vel - self.body_vel.reshape(-1,3*self.num_bodies)), dim=1)
        return torch.exp(-0.1 * diff_local_body_vel), diff_local_body_vel
    
    def _reward_local_ang_vel_phc(self):
        target_local_ang_vel = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                self.target_global_ang_vel.reshape(-1,self.num_bodies,3)).reshape(-1, 3 * self.num_bodies)
        diff_local_body_ang_vel = torch.mean(torch.square(target_local_ang_vel - self.body_ang_vel.reshape(-1,3*self.num_bodies)), dim=1)
        return torch.exp(-0.1 * diff_local_body_ang_vel), diff_local_body_ang_vel
    
    