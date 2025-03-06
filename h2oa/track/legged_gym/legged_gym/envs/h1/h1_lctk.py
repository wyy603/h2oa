import time
from warnings import WarningMessage
import numpy as np
import os
from copy import deepcopy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from scipy.spatial.transform import Rotation
# from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.diff_quat import flip_quat_by_w, quat_to_vec6d, quat_multiply, quat_inv, broadcast_quat_apply, broadcast_quat_multiply, vec6d_to_quat
from legged_gym.utils import diff_rot
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain
from legged_gym.privileged.observation import HistoryBuffer
from legged_gym.envs.h1.h1_lctk_config import H1RoughCfg
# from legged_gym.utils.gait.gait_planner import BipedalGaitPlanner
from legged_gym.utils.human import load_target_pkl_concat


EPS=1e-4

def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x) + 1 if np.random.rand() < (x - int(x)) else int(x)

class H1Walk(BaseTask):
    def __init__(self, cfg: H1RoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
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
        if self.debug_viz:
            self.test_motion_id = 0.
        self.init_done = False
        self.iteration = 0
        self.scaling_factor = 0.1
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        # human ref motion
        self._init_target_pkls()
        self.init_done = True



    def _init_target_pkls(self):
        self.target_jt_seq, self.target_global_seq, self.target_seq_len, self.motion_start_id = load_target_pkl_concat(self.device, self.cfg.human.filename)
        self.target_jt_seq = torch.concat([self.target_jt_seq, 
                                           torch.concat([self.default_dof_pos,self.default_dof_pos],dim=-1)], 
                                           dim=0)
        self.num_target_seq = self.target_seq_len.shape[0]
        print(f"Loaded target joint trajectories of shape {self.target_seq_len.shape}")
        # assert(self.dim_target_jt == self.num_dofs)
        if self.debug_viz:
            self.target_jt_i = torch.arange(self.num_envs, device=self.device) % self.num_target_seq
        else:
            self.target_jt_i = torch.randint(0, self.num_target_seq, (self.num_envs,), device=self.device)
        self.target_jt_j = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rand_still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_jt_dt = 1 / self.cfg.human.freq
        self.target_jt_update_steps = self.target_jt_dt / self.dt # not necessary integer
        assert(self.dt <= self.target_jt_dt)
        self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)

        # self.target_jt_pos = None
        # self.target_jt_vel = None

        # self.target_body_pos = None
        # self.target_body_ori = None
        # self.target_body_vel = None
        # self.target_body_ang_vel = None

        # self.target_root_pos = None
        # self.target_root_ori = None
        # self.target_root_vel = None
        # self.target_root_ang_vel = None

        # self.target_jt_steps = self.cfg.human.delay / self.target_jt_dt
        # self.target_jt_steps_int = sample_int_from_float(self.target_jt_steps)
        self.update_target_pkls(torch.tensor([], dtype=torch.long, device=self.device))

    def update_target_pkls(self, reset_env_ids):
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        resample_i[reset_env_ids] = True
        jt_eps_end_bool = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter >= self.target_jt_update_steps_int:
            self.common_step_counter = 0
            self.target_jt_j += 1
            jt_eps_end_bool = self.target_jt_j >= self.target_seq_len[self.target_jt_i]
            resample_i = resample_i | jt_eps_end_bool
            self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)
            # self.target_jt_steps_int = sample_int_from_float(self.target_jt_steps)

        if self.cfg.commands.loco_command:
            self._resample_commands(resample_i.nonzero(as_tuple=False).flatten())
        stance_mask = self.get_command_stop_mask() # vel too small
        rand_still = torch.rand(self.num_envs, device=self.device) < self.cfg.human.rand_still
        rand_still&= stance_mask
        self.rand_still = torch.where(resample_i, rand_still, self.rand_still)

        if hasattr(self.cfg, 'recycle_data') and self.cfg.recycle_data:

            obs_global_pos = (self.body_pos_raw - self.env_origins[:,None]).detach().cpu().clone().reshape(self.num_envs,-1)
            obs_global_ori = quat_to_vec6d(self.body_ori_raw).detach().cpu().clone().reshape(self.num_envs,-1)
            obs_global_vel = self.body_vel_raw.detach().cpu().clone().reshape(self.num_envs,-1)
            obs_global_ang_vel = self.body_ang_vel_raw.detach().cpu().clone().reshape(self.num_envs,-1)
            self.extras['obs_global_buf'] = torch.cat([obs_global_pos, obs_global_ori, obs_global_vel, obs_global_ang_vel], dim=1) # NOTE(xh) IMPORTANT BUG!!!!!!!!!!!! reshape first
            self.extras['obs_jt_buf'] = torch.cat([self.dof_pos,self.dof_vel], dim=1).detach().cpu().clone()

        # if self.cfg.human.resample_on_env_reset:
        self.extras['succ_id'] = self.target_jt_i[jt_eps_end_bool]
        if self.debug_viz:
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

        # print(self.target_jt_i)
        target_jt_i_cpu = self.target_jt_i#.to('cpu')
        target_jt_j_cpu = self.target_jt_j#.to('cpu')
        cum_id = self.motion_start_id[target_jt_i_cpu]+target_jt_j_cpu
        cum_id[self.rand_still] = -1

        # self.target_jt = torch.stack([self.target_jt_seq[i][j] for i, j in zip(target_jt_i_cpu, target_jt_j_cpu)], dim=0).to(self.device)
        self.target_jt = self.target_jt_seq[cum_id]#.to(self.device)
        self.target_jt_pos, self.target_jt_vel = self.target_jt[:,:19], self.target_jt[:,19:]

        # self.target_global = torch.stack([self.target_global_seq[i][j] for i, j in zip(target_jt_i_cpu, target_jt_j_cpu)], dim=0).to(self.device)
        self.target_global = self.target_global_seq[cum_id]#.to(self.device)
        self.target_global_pos, self.target_global_ori, self.target_global_vel, self.target_global_ang_vel = \
            self.target_global[:,:3*self.num_bodies], self.target_global[:,3*self.num_bodies:9*self.num_bodies], self.target_global[:,9*self.num_bodies:12*self.num_bodies], self.target_global[:,12*self.num_bodies:15*self.num_bodies]
        self.target_heading_quat_inv = diff_rot.calc_heading_quat_inv(vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)))[:,None]


        # delayed_target_jt_j_cpu = torch.maximum(target_jt_j_cpu - self.target_jt_steps_int, torch.tensor(0))

        # # self.target_jt = torch.stack([self.target_jt_seq[i][j] for i, j in zip(target_jt_i_cpu, delayed_target_jt_j_cpu)], dim=0).to(self.device)
        # self.target_jt = self.target_jt_seq[self.motion_start_id[target_jt_i_cpu]+delayed_target_jt_j_cpu]#.to(self.device)
        # self.target_jt_pos, self.target_jt_vel = self.target_jt[:,:19], self.target_jt[:,19:]
   
        # # self.target_global = torch.stack([self.target_global_seq[i][j] for i, j in zip(target_jt_i_cpu, delayed_target_jt_j_cpu)], dim=0).to(self.device)
        # self.target_global = self.target_global_seq[self.motion_start_id[target_jt_i_cpu]+delayed_target_jt_j_cpu]#.to(self.device)
        # self.target_global_pos, self.target_global_ori, self.target_global_vel, self.target_global_ang_vel =\
        #     self.target_global[:,:3*self.num_bodies], self.target_global[:,3*self.num_bodies:9*self.num_bodies], self.target_global[:,9*self.num_bodies:12*self.num_bodies], self.target_global[:,12*self.num_bodies:15*self.num_bodies]
        # self.target_heading_quat_inv = diff_rot.calc_heading_quat_inv(vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)))[:,None]

   
        # self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf = jt_eps_end_bool
        self.reset_buf = self.reset_buf | self.time_out_buf
        if 'termination' in self.extras:
            self.extras["termination"]['time_out_buf'] = self.time_out_buf
                    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            if self.cfg.domain_rand.randomize_proprio_latency:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.base_quat = self.root_states[:, 3:7]
                self.heading_quat_inv = diff_rot.calc_heading_quat_inv(self.base_quat)[:,None]
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                # self.base_ang_vel[:] = broadcast_quat_apply(self.heading_quat_inv.squeeze(1), self.root_states[:, 10:13]) # NOTE base_quat -> heading_quat_inv
                self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                self.update_proprio_latency_buf()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)


        self.episode_length_buf += 1
        self.common_step_counter += 1
        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.heading_quat_inv[:] = diff_rot.calc_heading_quat_inv(self.base_quat)[:,None]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_lin_vel[:] = broadcast_quat_apply(self.heading_quat_inv.squeeze(1), self.root_states[:, 7:10]) # NOTE base_quat -> heading_quat_inv
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.base_ang_vel[:] = broadcast_quat_apply(self.heading_quat_inv.squeeze(1), self.root_states[:, 10:13]) # NOTE base_quat -> heading_quat_inv
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # self.compute_local_foot_pos_and_vel()
        # TODO check every terms used in reward

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.update_target_pkls(env_ids)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._post_physics_step_callback(env_ids)
        self.reset_idx(env_ids) # Will body info updated here?


        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:].clone()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync: # and self.debug_viz
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.contact_reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.rpy_reset_buf = torch.logical_or(torch.abs(self.rpy[:, 1]) > self.cfg.env.termination_pitch, torch.abs(self.rpy[:, 0]) > self.cfg.env.termination_roll)
        # self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = self.contact_reset_buf | self.rpy_reset_buf

        self.extras["termination"] = {}
        self.extras["termination"]['termination_contact_buf'] = self.contact_reset_buf
        self.extras["termination"]['r_threshold_buff'] = self.rpy_reset_buf
        self.extras["termination"]['p_threshold_buff'] = self.rpy_reset_buf
        self.extras["termination"]['z_threshold_buff'] = self.rpy_reset_buf
        self.extras["termination"]['has_fallen'] = self.reset_buf # TODO
        # self.extras["termination"]['time_out_buf'] = self.time_out_buf

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

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # if self.cfg.commands.loco_command:
        #     self._resample_commands(env_ids)
        # if self.skill_curriculum:
        #     self.update_skill_curriculum(env_ids)

        # reset buffers
        self.rpy[env_ids] = get_euler_xyz_in_tensor(self.base_quat[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        if self.cfg.domain_rand.randomize_proprio_latency:
            self.angel_vel_latency_buf.reset_idx(env_ids, self.base_ang_vel[env_ids])
            self.projected_gravity_latency_buf.reset_idx(env_ids, self.projected_gravity[env_ids])
            self.dof_pos_latency_buf.reset_idx(env_ids, self.dof_pos[env_ids])
            self.dof_vel_latency_buf.reset_idx(env_ids, self.dof_vel[env_ids])

        # fill extras
        self.extras['episode_length_current'] = self.episode_length_buf.float().mean()
        self.extras['scaling_factor'] = self.scaling_factor
        self.extras["episode_metrics"] = deepcopy(self.episode_metrics)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # if self.cfg.commands.command_curriculum:
        #     self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # if self.cfg.commands.skill_curriculum:
        #     self.extras["episode"]["mean_skill_curriculum"] = torch.mean(self.skill_curriculum_idx)
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # self.gait_planner.reset_idx(env_ids)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            unscaled_rew, metric = self.reward_functions[i]()
            rew = unscaled_rew * self.reward_scales[name]
            if 'local_pos_phc' in name:
                rew *=0.
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

    def update_proprio_latency_buf(self):
        self.angel_vel_latency_buf.update(self.base_ang_vel)
        self.projected_gravity_latency_buf.update(self.projected_gravity)
        self.dof_pos_latency_buf.update(self.dof_pos)
        self.dof_vel_latency_buf.update(self.dof_vel)

    def compute_observations(self):
        """ Computes observations
        """
        if not self.cfg.domain_rand.randomize_proprio_latency:
                
            projected_gravity =self.projected_gravity
            base_ang_vel = self.base_ang_vel
            dof_pos = self.dof_pos
            dof_vel = self.dof_vel
            
        else:
            simsteps = self.proprio_latency_simsteps.unsqueeze(1).unsqueeze(2)
            projected_gravity = torch.gather(
                self.projected_gravity_latency_buf.get_history_buffer(),
                1, simsteps.expand(-1, -1, 3)).squeeze(1)
            base_ang_vel = torch.gather(self.angel_vel_latency_buf.get_history_buffer(),
                                        1, simsteps.expand(-1, -1, 3)).squeeze(1)
            dof_pos = torch.gather(self.dof_pos_latency_buf.get_history_buffer(),
                                   1, simsteps.expand(-1, -1, self.num_dofs)).squeeze(1)
            dof_vel = torch.gather(self.dof_vel_latency_buf.get_history_buffer(),
                                   1, simsteps.expand(-1, -1, self.num_dofs)).squeeze(1)

        if self.cfg.commands.loco_command:
            self.target_xy_vel = self.commands[...,:2]
            self.target_yaw_vel = self.commands[...,2:]
        else:
            # TODO not heading frame!!!!!!
            target_root_local_vel = broadcast_quat_apply(self.target_heading_quat_inv.squeeze(1), 
                                                self.target_global_vel[:,:3])
            target_root_local_ang_vel = broadcast_quat_apply(self.target_heading_quat_inv.squeeze(1), 
                                                self.target_global_ang_vel[:,:3])
            self.target_xy_vel = target_root_local_vel[...,:2]
            self.target_yaw_vel = target_root_local_ang_vel[...,2:]
        target_projected_gravity = quat_rotate_inverse(
            vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)), self.gravity_vec)
        cmd_scale=self.obs_scales.cmd
        # cmd_scale0=(self.episode_length_buf/80.).clamp_max(1.)
        # cmd_scale1=((250-self.episode_length_buf)/30.).clamp_max(1.)
        # cmd_scale=torch.min(cmd_scale0,cmd_scale1).clamp_min(0.)[:,None]
        # breakpoint()
        self.obs_buf = torch.cat((
            # self.commands,  # 3
            (self.target_jt_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.target_jt_vel * self.obs_scales.dof_vel,
            (self.target_jt_pos-dof_pos) * self.obs_scales.dof_pos,
            (self.target_jt_vel-dof_vel) * self.obs_scales.dof_vel,
            target_projected_gravity,
            self.target_xy_vel * cmd_scale,  # 3 TODO lin xy ang yaw; align with reward
            self.target_yaw_vel * cmd_scale,  # 3 TODO lin xy ang yaw; align with reward
            projected_gravity,  # 3
            base_ang_vel * self.obs_scales.ang_vel,  # 3
            (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 10
            dof_vel * self.obs_scales.dof_vel,  # 10
            self.actions,  # 10
            # gait command
            # self.gait_planner.gait_indices.unsqueeze(1),  # 1
            # self.gait_planner.clock_inputs  # 2
        ), dim=-1)

        body_height = self.base_pos[:, 2].unsqueeze(1)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)

        foot_contact_force = self.force_sensor_tensor[:, :, 2]
        foot_contact_state = self.force_sensor_tensor.norm(dim=-1) > 1.5

        self.privileged_obs_buf = torch.cat(( # TODO add global
            # privileged
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.obs_buf.clone(),  # proprioceptive info
            self.friction_coeffs_tensor,  # 1
            self.mass_params_tensor,  # 4
            self.motor_strength - 1,  # 12
            self.foot_velocities.view(self.num_envs, -1),  # 6
            foot_contact_force / 10,  # 2
            foot_contact_state,  # 2
            body_height,  # 1
            foot_height,  # 2
            # terrain_heights, # only with vision
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        print("Added ground plane with friction: {}, restitution: {}".format(plane_params.static_friction,
                                                                             plane_params.restitution))

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
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

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
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        self._create_terrain()
        self._create_envs()
        print("Simulation created.")

    def _create_terrain(self):
        start_time = time.time()
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type in ['trimesh', 'heightfield']:
            self.terrain = Terrain(self.cfg.terrain)
            if mesh_type == 'trimesh':
                self._create_trimesh()
            elif mesh_type == 'heightfield':
                self._create_heightfield()
            else:
                raise ValueError(f"Unknown terrain type: {mesh_type}")

        else:
            raise ValueError(f"Unknown terrain type: {mesh_type}")
        print(f"Terrain created in {time.time() - start_time:.2f} s")

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
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
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
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

    def _post_physics_step_callback(self, env_ids):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        # env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
        #     as_tuple=False).flatten()

        if self.cfg.domain_rand.push_robots and (int(self.episode_length_buf[0]) % self.cfg.domain_rand.push_interval == 0): # NOTE
            self._push_robots() 

        # stop_mask = self.get_command_stop_mask()
        # self.gait_planner.update_gait_phase(stop_mask)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
    
        commands_list = []
        if self.cfg.commands.loco_command:
            self.loco_commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                              self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                              device=self.device).squeeze(1)
            self.loco_commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                              self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                              device=self.device).squeeze(1)
            self.loco_commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                              self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                              device=self.device).squeeze(1)
            # set small commands to zero
            self.loco_commands[env_ids, :2] *= torch.abs(self.loco_commands[env_ids, :2]) >= self.cfg.commands.lin_vel_clip
            self.loco_commands[env_ids, 2] *= torch.abs(self.loco_commands[env_ids, 2]) >= self.cfg.commands.ang_vel_clip

            # self.loco_commands[env_ids, :2] *= (torch.norm(self.loco_commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
            # self.loco_commands[env_ids, 2] *= (torch.abs(self.loco_commands[env_ids, 2]) > 0.3)
        if self.cfg.commands.hand_touch_command:
            self.left_hand_touch_commands[env_ids, 0] = torch_rand_float(self.command_ranges["touch_x_left"][0],
                                                                    self.command_ranges["touch_x_left"][1],
                                                                    (len(env_ids), 1),
                                                                    device=self.device).squeeze(1)
            self.left_hand_touch_commands[env_ids, 1] = torch_rand_float(self.command_ranges["touch_y_left"][0],
                                                                    self.command_ranges["touch_y_left"][1],
                                                                    (len(env_ids), 1),
                                                                    device=self.device).squeeze(1)
            self.left_hand_touch_commands[env_ids, 2] = torch_rand_float(self.command_ranges["touch_z_left"][0],
                                                                    self.command_ranges["touch_z_left"][1],
                                                                    (len(env_ids), 1),
                                                                    device=self.device).squeeze(1)

            self.right_hand_touch_commands[env_ids, 0] = torch_rand_float(self.command_ranges["touch_x_right"][0],
                                                                    self.command_ranges["touch_x_right"][1],
                                                                    (len(env_ids), 1),
                                                                    device=self.device).squeeze(1)
            self.right_hand_touch_commands[env_ids, 1] = torch_rand_float(self.command_ranges["touch_y_right"][0],
                                                                    self.command_ranges["touch_y_right"][1],
                                                                    (len(env_ids), 1),
                                                                    device=self.device).squeeze(1)
            self.right_hand_touch_commands[env_ids, 2] = torch_rand_float(self.command_ranges["touch_z_right"][0],
                                                                    self.command_ranges["touch_z_right"][1],
                                                                    (len(env_ids), 1),
                                                                    device=self.device).squeeze(1)

        if self.cfg.commands.squat_command:
            self.squat_commands[env_ids, 0] = torch_rand_float(self.command_ranges["squat_z"][0],
                                                               self.command_ranges["squat_z"][1], (len(env_ids), 1),
                                                               device=self.device).squeeze(1)
            
        if self.cfg.commands.pitch_command:
            self.pitch_commands[env_ids, 0] = torch_rand_float(self.command_ranges["pitch_forward"][0],
                                                               self.command_ranges["pitch_forward"][1], (len(env_ids), 1),
                                                               device=self.device).squeeze(1)
            
        if self.skill_curriculum:
            env_ids_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            env_ids_bool[env_ids] = True

            # single skill
            # loco_skill_bool = torch.logical_and(env_ids_bool, self.skill_curriculum_idx == self.skill_curriculum_list.indexfind("locomotion"))
            # squat_skill_bool = torch.logical_and(env_ids_bool, self.skill_curriculum_idx == self.skill_curriculum_list.index("squat_down"))
            # touch_point_skill_bool = torch.logical_and(env_ids_bool, self.skill_curriculum_idx == self.skill_curriculum_list.index("touch_point"))
            # loco_skill_mask_bool = torch.logical_or(squat_skill_bool, touch_point_skill_bool)
            # squat_skill_mask_bool = torch.logical_or(loco_skill_bool, touch_point_skill_bool)
            # touch_point_skill_mask_bool = torch.logical_or(loco_skill_bool, squat_skill_bool)

            # self.loco_commands[loco_skill_mask_bool] = self.loco_commands_mask[0]
            # self.left_hand_touch_commands[touch_point_skill_mask_bool] = self.left_hand_touch_commands_mask[0]
            # self.right_hand_touch_commands[touch_point_skill_mask_bool] = self.right_hand_touch_commands_mask[0]
            # self.squat_commands[squat_skill_mask_bool] = self.squat_commands_mask[0]
            
            # multi single skill
            command_prob = torch_rand_float(0, 3,(self.num_envs, 1),device=self.device).squeeze(1)
            loco_bool = torch.logical_and(command_prob >= 0, command_prob < 1)
            touch_point_bool = torch.logical_and(command_prob >= 1, command_prob < 2)
            squat_bool = torch.logical_and(command_prob >= 2, command_prob <= 3)
            multi_single_skill_bool = torch.logical_and(env_ids_bool, self.skill_curriculum_idx == self.skill_curriculum_list.index("multi_single_skill"))
            
            loco_single_skill_bool = torch.logical_and(loco_bool, multi_single_skill_bool)
            squat_single_skill_bool = torch.logical_and(squat_bool, multi_single_skill_bool)
            touch_point_single_skill_bool = torch.logical_and(touch_point_bool, multi_single_skill_bool)

            loco_single_skill_mask_bool = torch.logical_or(squat_single_skill_bool, touch_point_single_skill_bool)
            squat_single_skill_mask_bool = torch.logical_or(loco_single_skill_bool, touch_point_single_skill_bool)
            touch_point_single_skill_mask_bool = torch.logical_or(loco_single_skill_bool, squat_single_skill_bool)

            self.loco_commands[loco_single_skill_mask_bool] = self.loco_commands_mask[0]
            self.left_hand_touch_commands[touch_point_single_skill_mask_bool] = self.left_hand_touch_commands_mask[0]
            self.right_hand_touch_commands[touch_point_single_skill_mask_bool] = self.right_hand_touch_commands_mask[0]
            self.squat_commands[squat_single_skill_mask_bool] = self.squat_commands_mask[0]

        if self.commands_mask and (not self.skill_curriculum): # do not support pitch for now
            command_prob = torch_rand_float(0, 2,(self.num_envs, 1),device=self.device).squeeze(1)
            env_ids_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            env_ids_bool[env_ids] = True
            loco_bool = torch.logical_and(command_prob >= 0, command_prob < 1.4)
            manip_bool = torch.logical_and(command_prob >= 1.4, command_prob <= 2)
            loco_mask_bool = torch.logical_and(env_ids_bool, torch.logical_not(loco_bool))
            manip_mask_bool = torch.logical_and(env_ids_bool, torch.logical_not(manip_bool))
            self.loco_commands[loco_mask_bool] = self.loco_commands_mask[0]
            self.left_hand_touch_commands[manip_mask_bool] = self.left_hand_touch_commands_mask[0]
            self.right_hand_touch_commands[manip_mask_bool] = self.right_hand_touch_commands_mask[0]
            self.squat_commands[manip_mask_bool] = self.squat_commands_mask[0]

        if self.command_highway and self.iteration < self.command_stophighway + 5:
            # highway_prob = torch_rand_float(0, 2,(self.num_envs, 1),device=self.device).squeeze(1)
            # high_highway_bool = highway_prob > 1.8
            # low_highway_bool = highway_prob < 1.0
            # env_ids_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # env_ids_bool[env_ids] = True
            # set_high_highway_bool = torch.logical_and(high_highway_bool, env_ids_bool)
            # set_low_highway_bool = torch.logical_and(low_highway_bool, env_ids_bool)
            # # set highway
            # self.squat_commands[set_low_highway_bool] = self.command_ranges["squat_z"][0]
            # self.squat_commands[set_high_highway_bool] = self.command_ranges["squat_z"][1]

            command_prob = torch_rand_float(0, 2,(self.num_envs, 1),device=self.device).squeeze(1)
            env_ids_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            env_ids_bool[env_ids] = True
            loco_bool = torch.logical_and(command_prob >= 0, command_prob < 1)
            manip_bool = torch.logical_and(command_prob >= 1, command_prob <= 2)
            loco_mask_bool = torch.logical_and(env_ids_bool, torch.logical_not(loco_bool))
            manip_mask_bool = torch.logical_and(env_ids_bool, torch.logical_not(manip_bool))
            self.loco_commands[loco_mask_bool] = self.loco_commands_mask[0]
            self.left_hand_touch_commands[manip_mask_bool] = self.left_hand_touch_commands_mask[0]
            self.right_hand_touch_commands[manip_mask_bool] = self.right_hand_touch_commands_mask[0]
            self.squat_commands[manip_mask_bool] = self.squat_commands_mask[0]

        if self.command_teacher:
            env_ids_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            env_ids_bool[env_ids] = True
            change_bool = torch.logical_and(self.command_teacher_bool[:, 0], env_ids_bool)
            up_down_prob = torch_rand_float(0, 2,(self.num_envs, 1),device=self.device).squeeze(1)
            down_bool = up_down_prob < 1.0
            up_bool = up_down_prob >= 1.0
            down_change_bool = torch.logical_and(down_bool, change_bool)
            up_change_bool = torch.logical_and(up_bool, change_bool)
            self.squat_commands[down_change_bool] = torch.clip(self.last_squat_commands[down_change_bool] - 0.1, self.command_ranges["squat_z"][0], self.command_ranges["squat_z"][1])
            self.squat_commands[up_change_bool] = torch.clip(self.last_squat_commands[up_change_bool] + 0.1, self.command_ranges["squat_z"][0], self.command_ranges["squat_z"][1])
            self.last_squat_commands[down_change_bool] = self.squat_commands[down_change_bool]
            self.last_squat_commands[up_change_bool] = self.squat_commands[up_change_bool]
 
        if self.cfg.commands.loco_command:
            commands_list.append(self.loco_commands)
        if self.cfg.commands.hand_touch_command:
            commands_list.append(self.left_hand_touch_commands)
        if self.cfg.commands.hand_touch_command:
            commands_list.append(self.right_hand_touch_commands)
        if self.cfg.commands.squat_command:    
            commands_list.append(self.squat_commands)
        if self.cfg.commands.pitch_command:
            commands_list.append(self.pitch_commands)
        self.commands = torch.cat(commands_list, dim=-1)

        if self.cfg.domain_rand.reset_arm_dof_when_resampling_command and len(env_ids) > 0:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dofs),
                                                                        device=self.device)
            dof_pos_noise = torch_rand_float(-np.deg2rad(20), np.deg2rad(20),
                                         (len(env_ids), self.num_dofs), device=self.device)
            self.dof_pos[env_ids] += dof_pos_noise

            dof_pelvis_noise = torch_rand_float(-np.deg2rad(30), np.deg2rad(30),
                                         (len(env_ids), 1), device=self.device)
            self.dof_pos[env_ids, 10:11] += dof_pelvis_noise

            dof_vel_noise = torch_rand_float(-0.5, 0.5,
                                         (len(env_ids), self.num_dofs), device=self.device)
            self.dof_vel[env_ids] = 0.
            self.dof_vel[env_ids] += dof_vel_noise

            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled *= self.motor_strength
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (
                    actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
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
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dofs),
                                                                        device=self.device)
        dof_pos_noise = torch_rand_float(-np.deg2rad(20), np.deg2rad(20),
                                         (len(env_ids), self.num_dofs), device=self.device)
        self.dof_pos[env_ids] += dof_pos_noise

        # dof_pelvis_noise = torch_rand_float(-np.deg2rad(30), np.deg2rad(30),
        #                                  (len(env_ids), 1), device=self.device)
    
        # self.dof_pos[env_ids, 10:11] += dof_pelvis_noise

        dof_vel_noise = torch_rand_float(-0.5, 0.5,
                                         (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.
        self.dof_vel[env_ids] += dof_vel_noise

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # TODO only in trimesh
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] = self.target_global_pos[env_ids,:3]
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids,3:7] = diff_rot.calc_heading_quat(vec6d_to_quat(self.target_global_ori[env_ids,:6].reshape(-1,3,2)))
            # self.root_states[env_ids] = self.base_init_state
            # self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # self.root_states[env_ids, :2] += torch_rand_float(-3.0, 3.0, (len(env_ids), 2),
            #                                                   device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            
        # base orientations
        # rand_ori = torch.zeros((len(env_ids), 3), device=self.device)
        # rand_pitch = torch_rand_float(0., 0.3, (len(env_ids), 1), device=self.device)
        # rand_ori[:,1:2] = rand_pitch
        # rand_quat = matrix_to_quaternion(euler_angles_to_matrix(rand_ori, 'XYZ'))
        # rand_quat_w = rand_quat[:,:1].clone()
        # rand_quat_xyz = rand_quat[:,1:].clone()
        # rand_quat[:,:3] = rand_quat_xyz
        # rand_quat[:,3:] = rand_quat_w
        # rand_quat = flip_quat_by_w(rand_quat)
        # self.root_states[env_ids, 3:7] = rand_quat

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                                                    device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        if self.cfg.domain_rand.reset_arm_dof_when_push_robots:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dofs),
                                                                        device=self.device)
            dof_pos_noise = torch_rand_float(-np.deg2rad(20), np.deg2rad(20),
                                         (len(env_ids), self.num_dofs), device=self.device)
            self.dof_pos[env_ids] += dof_pos_noise

            dof_pelvis_noise = torch_rand_float(-np.deg2rad(30), np.deg2rad(30),
                                         (len(env_ids), 1), device=self.device)
            self.dof_pos[env_ids, 10:11] += dof_pelvis_noise

            dof_vel_noise = torch_rand_float(-0.5, 0.5,
                                         (len(env_ids), self.num_dofs), device=self.device)
            self.dof_vel[env_ids] = 0.
            self.dof_vel[env_ids] += dof_vel_noise

            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def update_command_curriculum(self):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # strategy 1
        # if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
        #     self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, self.max_command_curriculum_ranges["lin_vel_x"][0], 0.)
        #     self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.max_command_curriculum_ranges["lin_vel_x"][1])
        #     self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.2, self.max_command_curriculum_ranges["lin_vel_y"][0], 0.)
        #     self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.2, 0., self.max_command_curriculum_ranges["lin_vel_y"][1])

        # if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_ang_vel"]:
        #     self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.2, self.max_command_curriculum_ranges["ang_vel_yaw"][0], 0.)
        #     self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.2, 0., self.max_command_curriculum_ranges["ang_vel_yaw"][1])

        # if torch.mean((self.episode_sums["tracking_left_hand"][env_ids] + self.episode_sums["tracking_right_hand"][env_ids]) / 2) / self.max_episode_length > 0.8 * self.reward_scales["tracking_left_hand"]:
        #     self.command_ranges["touch_x_left"][0] = np.clip(self.command_ranges["touch_x_left"][0] - 0.2, self.max_command_curriculum_ranges["touch_x_left"][0], 100.)
        #     self.command_ranges["touch_x_left"][1] = np.clip(self.command_ranges["touch_x_left"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_x_left"][1])
        #     self.command_ranges["touch_y_left"][0] = np.clip(self.command_ranges["touch_y_left"][0] - 0.2, self.max_command_curriculum_ranges["touch_y_left"][0], 100.)
        #     self.command_ranges["touch_y_left"][1] = np.clip(self.command_ranges["touch_y_left"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_y_left"][1])
        #     self.command_ranges["touch_z_left"][0] = np.clip(self.command_ranges["touch_z_left"][0] - 0.2, self.max_command_curriculum_ranges["touch_z_left"][0], 100.)
        #     self.command_ranges["touch_z_left"][1] = np.clip(self.command_ranges["touch_z_left"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_z_left"][1])

        #     self.command_ranges["touch_x_right"][0] = np.clip(self.command_ranges["touch_x_right"][0] - 0.2, self.max_command_curriculum_ranges["touch_x_right"][0], 100.)
        #     self.command_ranges["touch_x_right"][1] = np.clip(self.command_ranges["touch_x_right"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_x_right"][1])
        #     self.command_ranges["touch_y_right"][0] = np.clip(self.command_ranges["touch_y_right"][0] - 0.2, self.max_command_curriculum_ranges["touch_y_right"][0], 100.)
        #     self.command_ranges["touch_y_right"][1] = np.clip(self.command_ranges["touch_y_right"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_y_right"][1])
        #     self.command_ranges["touch_z_right"][0] = np.clip(self.command_ranges["touch_z_right"][0] - 0.2, self.max_command_curriculum_ranges["touch_z_right"][0], 100.)
        #     self.command_ranges["touch_z_right"][1] = np.clip(self.command_ranges["touch_z_right"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_z_right"][1])

        # strategy 2 
        # self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, self.max_command_curriculum_ranges["lin_vel_x"][0], 0.)
        # self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.max_command_curriculum_ranges["lin_vel_x"][1])
        # self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.2, self.max_command_curriculum_ranges["lin_vel_y"][0], 0.)
        # self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.2, 0., self.max_command_curriculum_ranges["lin_vel_y"][1])

        # self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.2, self.max_command_curriculum_ranges["ang_vel_yaw"][0], 0.)
        # self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.2, 0., self.max_command_curriculum_ranges["ang_vel_yaw"][1])

        # self.command_ranges["touch_x_left"][0] = np.clip(self.command_ranges["touch_x_left"][0] - 0.2, self.max_command_curriculum_ranges["touch_x_left"][0], 100.)
        # self.command_ranges["touch_x_left"][1] = np.clip(self.command_ranges["touch_x_left"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_x_left"][1])
        # self.command_ranges["touch_y_left"][0] = np.clip(self.command_ranges["touch_y_left"][0] - 0.2, self.max_command_curriculum_ranges["touch_y_left"][0], 100.)
        # self.command_ranges["touch_y_left"][1] = np.clip(self.command_ranges["touch_y_left"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_y_left"][1])
        # self.command_ranges["touch_z_left"][0] = np.clip(self.command_ranges["touch_z_left"][0] - 0.2, self.max_command_curriculum_ranges["touch_z_left"][0], 100.)
        # self.command_ranges["touch_z_left"][1] = np.clip(self.command_ranges["touch_z_left"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_z_left"][1])

        # self.command_ranges["touch_x_right"][0] = np.clip(self.command_ranges["touch_x_right"][0] - 0.2, self.max_command_curriculum_ranges["touch_x_right"][0], 100.)
        # self.command_ranges["touch_x_right"][1] = np.clip(self.command_ranges["touch_x_right"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_x_right"][1])
        # self.command_ranges["touch_y_right"][0] = np.clip(self.command_ranges["touch_y_right"][0] - 0.2, self.max_command_curriculum_ranges["touch_y_right"][0], 100.)
        # self.command_ranges["touch_y_right"][1] = np.clip(self.command_ranges["touch_y_right"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_y_right"][1])
        # self.command_ranges["touch_z_right"][0] = np.clip(self.command_ranges["touch_z_right"][0] - 0.2, self.max_command_curriculum_ranges["touch_z_right"][0], 100.)
        # self.command_ranges["touch_z_right"][1] = np.clip(self.command_ranges["touch_z_right"][1] + 0.2, -100., self.max_command_curriculum_ranges["touch_z_right"][1])

        # self.command_ranges["squat_z"][0] = np.clip(self.command_ranges["squat_z"][0] - 0.2, self.max_command_curriculum_ranges["squat_z"][0], 100.)
        # self.command_ranges["squat_z"][1] = np.clip(self.command_ranges["squat_z"][1] + 0.2, -100., self.max_command_curriculum_ranges["squat_z"][1])

        # strategy 3
        self.command_curriculum_idx += 1
        self.command_ranges["squat_z"][0] = self.command_curriculum_list[self.command_curriculum_idx][0]
        self.command_ranges["squat_z"][1] = self.command_curriculum_list[self.command_curriculum_idx][1]
            
            
    def update_skill_curriculum(self, env_ids):
        # strategy 1
        if self.skill_curriculum_strategy == 1:
            skill_curriculum_idx = self.skill_curriculum_idx[env_ids]
            for i, skill in enumerate(self.skill_curriculum_list):
                mean_rew = torch.zeros(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False)
                for j, rew in enumerate(self.skill_rew_map[skill]):
                    mean_rew += ((self.episode_sums[rew][env_ids] / self.max_episode_length) / self.reward_scales[rew]) * (skill_curriculum_idx == i)
                mean_rew = mean_rew / len(self.skill_rew_map[skill])
                self.skill_curriculum_idx[env_ids] += 1 * (mean_rew > 0.8) * (self.skill_curriculum_idx[env_ids] < (len(self.skill_curriculum_list) - 1)) * (skill_curriculum_idx == i)

        # strategy 2
        if self.skill_curriculum_strategy == 2:
            return 
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        obs_noise_scale_dim = [
            (4, 0.),  # 3
            (3, noise_scales.gravity),  # 3
            (3, noise_scales.ang_vel * self.obs_scales.ang_vel),  # 3
            (self.num_actions, noise_scales.dof_pos * self.obs_scales.dof_pos),  # 19
            (self.num_actions, noise_scales.dof_vel * self.obs_scales.dof_vel),  # 19
            (self.num_actions, 0.)  # 19
        ]
        for i, noise_scale_dim in enumerate(obs_noise_scale_dim):
            start_dim = sum([x[0] for x in obs_noise_scale_dim[:i]])
            end_dim = start_dim + noise_scale_dim[0]
            noise_vec[start_dim: end_dim] = noise_level * noise_scale_dim[1]

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.force_sensor_tensor = gymtorch.wrap_tensor(
            force_sensor_tensor).view(self.num_envs, len(self.force_sensor_dict), 6)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(
            self.num_envs, self.num_bodies, 13)  # + 1 self.body_state in xh
        self.rigid_body_state = self._rigid_body_state[:, :, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.body_pos_raw = self.rigid_body_state[..., :3]
        self.body_ori_raw = self.rigid_body_state[..., 3:7]
        self.body_vel_raw = self.rigid_body_state[..., 7:10]
        self.body_ang_vel_raw = self.rigid_body_state[..., 10:]
        self.base_quat = self.root_states[:, 3:7]
        self.heading_quat = diff_rot.calc_heading_quat(self.base_quat)[:,None]
        self.heading_quat_inv = diff_rot.calc_heading_quat_inv(self.base_quat)[:,None]

        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(
            net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.num_commands = 0
        self.target_xy_vel = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.target_yaw_vel = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        if self.cfg.commands.loco_command: # TODO
            self.num_commands += 3
            self.loco_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
            self.loco_commands_mask = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
            self.last_loco_commands = self.loco_commands_mask.clone()
        if self.cfg.commands.heading_command:
            self.num_commands += 1
            self.heading_commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        if self.cfg.commands.hand_touch_command:
            self.num_commands += 6
            self.left_hand_touch_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
            self.right_hand_touch_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
            self.left_hand_touch_commands_mask = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
            self.right_hand_touch_commands_mask = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
            self.left_hand_touch_commands_mask[:] = torch.tensor([0.33, 0.215, 0.08])
            self.right_hand_touch_commands_mask[:] = torch.tensor([0.33, -0.215, 0.08])
            self.last_left_hand_touch_commands = self.left_hand_touch_commands_mask.clone()
            self.last_right_hand_touch_commands = self.right_hand_touch_commands_mask.clone()
        if self.cfg.commands.squat_command:
            self.num_commands += 1
            self.squat_commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.squat_commands_mask = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.squat_commands_mask[:] = torch.tensor([0.98])
            self.last_squat_commands = self.squat_commands_mask.clone() - 0.28
        if self.cfg.commands.pitch_command:
            self.num_commands += 1
            self.pitch_commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.pitch_commands_mask = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.last_pitch_commands = self.pitch_commands_mask.clone()

        self.commands = torch.zeros(self.num_envs, self.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.commands_scale = torch.ones(self.num_commands,
                                         device=self.device, requires_grad=False)
        
        self.commands_mask = self.cfg.commands.mask

        self.command_curriculum = self.cfg.commands.command_curriculum
        self.command_curriculum_interval = self.cfg.commands.command_curriculum_interval
        self.command_curriculum_idx = 0
        # self.command_curriculum_list = self.cfg.commands.command_curriculum_list
        self.command_highway = self.cfg.commands.command_highway
        self.command_stophighway = self.cfg.commands.command_stophighway
        self.command_teacher = self.cfg.commands.command_teacher
        if self.command_teacher:
            self.command_teacher_bool = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)
            self.command_teacher_bool = self.command_teacher_bool * torch_rand_float(0, 1.0, (self.num_envs, 1), device=self.device) < self.cfg.commands.command_teacher_ratio

        self.skill_curriculum = self.cfg.commands.skill_curriculum
        self.skill_curriculum_strategy = self.cfg.commands.skill_curriculum_strategy
        self.skill_curriculum_interval = self.cfg.commands.skill_curriculum_interval
        self.skill_curriculum_list = self.cfg.commands.skill_curriculum_list
        self.skill_curriculum_loop = self.cfg.commands.skill_curriculum_loop
        self.skill_curriculum_idx = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.skill_curriculum_loop_idx = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.skill_rew_map = self.cfg.commands.skill_rew_map
        self.skill_rew_list = self.cfg.commands.skill_rew_list

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.ones(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.domain_rand.randomize_proprio_latency:
            low_late_simsteps = int(self.cfg.domain_rand.proprio_latency_range[0] / self.sim_params.dt)
            high_late_simsteps = int(self.cfg.domain_rand.proprio_latency_range[1] / self.sim_params.dt) + 1
            self.proprio_latency_simsteps = torch.randint(
                low=low_late_simsteps, high=high_late_simsteps, size=(self.num_envs,),
                device=self.device, dtype=torch.long)
            self.angel_vel_latency_buf = HistoryBuffer(self.num_envs, high_late_simsteps, 3, self.device)
            self.projected_gravity_latency_buf = HistoryBuffer(self.num_envs, high_late_simsteps, 3, self.device)
            self.dof_pos_latency_buf = HistoryBuffer(self.num_envs, high_late_simsteps, self.num_dofs, self.device)
            self.dof_vel_latency_buf = HistoryBuffer(self.num_envs, high_late_simsteps, self.num_dofs, self.device)

        # self.compute_local_foot_pos_and_vel()

        self.foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state[:, self.feet_indices, 0:3]

        # joint positions offsets and PD gains
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

        # self.gait_planner = BipedalGaitPlanner(self.num_envs,
        #                                        self.dt,
        #                                        self.device,
        #                                        self.cfg.control.gait.frequencies,
        #                                        stance_ratio=self.cfg.control.gait.stance_ratio,
        #                                        phase_offset=self.cfg.control.gait.phase_offset,
        #                                        kappa_gait_probs=self.cfg.control.gait.kappa_gait_probs)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
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
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # self.num_bodies = len(body_names)
        # self.num_dofs = len(self.dof_names)

        assert(self.num_bodies == len(body_names) and self.num_bodies == self.cfg.env.num_bodies)
        assert(self.num_dofs == len(self.dof_names))
        def any_in(patterns, string):
            return any(p in string for p in patterns)
        
        self.dof_hip_yaw_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'hip_yaw' in name],
                                                dtype=torch.long, device=self.device, requires_grad=False)
        
        self.dof_hip_pitch_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'hip_pitch' in name],
                                                dtype=torch.long, device=self.device, requires_grad=False)
        
        self.dof_hip_roll_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'hip_roll' in name],
                                                dtype=torch.long, device=self.device, requires_grad=False)

        self.dof_hip_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'hip' in name],
                                                dtype=torch.long, device=self.device, requires_grad=False)
        
        self.dof_ankle_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'ankle' in name],
                                              dtype=torch.long, device=self.device, requires_grad=False)

        self.dof_arm_indices = torch.tensor([i for i, name in enumerate(self.dof_names)
                                             if any_in(['shoulder', 'elbow'], name)],
                                            dtype=torch.long, device=self.device, requires_grad=False)

        self.dof_waist_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'torso' in name],
                                              dtype=torch.long, device=self.device, requires_grad=False)

        self.dof_knee_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'knee' in name],
                                              dtype=torch.long, device=self.device, requires_grad=False)
        
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        hand_names = [s for s in body_names if self.cfg.asset.hand_name in s]
        upper_dof_names = []
        for name in self.cfg.asset.upper_dof_names:
            upper_dof_names.extend([s for s in self.dof_names if name in s])
        upper_dof_names_no_left = []
        for name in self.cfg.asset.upper_dof_names_no_left:
            upper_dof_names_no_left.extend([s for s in self.dof_names if name in s])
        upper_dof_names_no_right = []
        for name in self.cfg.asset.upper_dof_names_no_right:
            upper_dof_names_no_right.extend([s for s in self.dof_names if name in s])
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

        # add force sensors
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
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
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

        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])
            
        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)   

        for i in range(len(hand_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         hand_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

        self.upper_dof_indices = torch.zeros(len(upper_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(upper_dof_names)):
            self.upper_dof_indices[i] = self.dof_names.index(upper_dof_names[i])

        self.upper_dof_indices_no_left = torch.zeros(len(upper_dof_names_no_left), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(upper_dof_names_no_left)):
            self.upper_dof_indices_no_left[i] = self.dof_names.index(upper_dof_names_no_left[i])

        self.upper_dof_indices_no_right = torch.zeros(len(upper_dof_names_no_right), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(upper_dof_names_no_right)):
            self.upper_dof_indices_no_right[i] = self.dof_names.index(upper_dof_names_no_right[i])
        all_dof_indices = torch.arange(self.num_dofs, device=self.device)
        self.lower_dof_indices = all_dof_indices[~torch.isin(all_dof_indices, self.upper_dof_indices)]

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
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
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
        # if not self.cfg.commands.command_curriculum:
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # if self.cfg.commands.command_curriculum:
        #     self.command_ranges = class_to_dict(self.cfg.commands.initial_curriculum_ranges)
        #     self.max_command_curriculum_ranges = class_to_dict(self.cfg.commands.ranges)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

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

    def get_command_stop_mask(self): # TODO

        # target_root_local_vel = broadcast_quat_apply(self.target_heading_quat_inv.squeeze(1), 
        #                                     self.target_global_vel[:,:3])
        # target_root_local_ang_vel = broadcast_quat_apply(self.target_heading_quat_inv.squeeze(1), 
        #                                     self.target_global_ang_vel[:,:3])
        lin_x_stop = torch.abs(self.target_xy_vel[:, 0]) < self.cfg.commands.lin_vel_clip
        lin_y_stop = torch.abs(self.target_xy_vel[:, 1]) < self.cfg.commands.lin_vel_clip
        ang_z_stop = torch.abs(self.target_yaw_vel[:, 0]) < self.cfg.commands.ang_vel_clip
        # lin_stop = torch.norm(self.loco_commands[:, :2], dim=1) <= self.cfg.commands.lin_vel_clip
        # ang_stop = torch.abs(self.loco_commands[:, 2]) <= self.cfg.commands.ang_vel_clip
        stop_mask = lin_x_stop & lin_y_stop & ang_z_stop
        # stop_mask = lin_stop & ang_stop
        return stop_mask # NOTE
    
    # def compute_local_foot_pos_and_vel(self):
    #     self.world_foot_positions = self.rigid_body_state[:, self.feet_indices, 0:3]
    #     self.world_foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]
    #     # base_to_world_matpose = quatpose_to_matpose(self.root_states[:, 0:7])
    #     # foot_to_world_matpose = quatpose_to_matpose(
    #     #     self.rigid_body_state[:, self.feet_indices, :7].view(-1, 7)).view(-1, len(self.feet_indices), 4, 4)

    #     # foot_to_base_matpose = torch.inverse(
    #     #     base_to_world_matpose.unsqueeze(1)) @ foot_to_world_matpose

    #     # # world_foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]
    #     # # world_foot_to_base_translation = self.world_foot_positions - self.root_states[:, 0:3].unsqueeze(1)
    #     # # self.local_foot_positions = (base_to_world_matpose[:, :3, :3].unsqueeze(1)
    #     # #                              @ world_foot_to_base_translation.unsqueeze(-1))
    #     # # self.local_foot_velocities = (base_to_world_matpose[:, :3, :3].unsqueeze(1)
    #     # #                               @ world_foot_velocities.unsqueeze(-1))
    #     # self.local_foot_quatpose = matpose_to_quatpose(
    #     #     foot_to_base_matpose.view(-1, 4, 4)).view(-1, len(self.feet_indices), 7)

    # ------------ reward functions----------------

    def _reward_target_jt_pos(self):
        # Penalize distance to target joint angles
        target_jt_pos = self.target_jt_pos#+self.default_dof_pos # NOTE(xh): add default pos to target jt
        target_jt_pos_error = torch.mean(torch.abs(self.dof_pos - target_jt_pos), dim=1)
        return torch.exp(-4 * target_jt_pos_error), target_jt_pos_error
    
    def _reward_target_updof(self):
        # Penalize distance to target joint angles
        target_jt_pos = self.target_jt_pos#+self.default_dof_pos # NOTE(xh): add default pos to target jt
        soft_error = torch.clamp_min(torch.abs(self.dof_pos - target_jt_pos) - 0.05,0.)
        target_jt_pos_error = torch.mean(soft_error[:,self.upper_dof_indices], dim=1)
        return torch.exp(-4 * target_jt_pos_error), target_jt_pos_error
    
    def _reward_target_lwdof(self):
        # Penalize distance to target joint angles
        target_jt_pos = self.target_jt_pos#+self.default_dof_pos # NOTE(xh): add default pos to target jtdof_hip_pitch_indices
        target_jt_pos_error = torch.mean(torch.abs(self.dof_pos - target_jt_pos)[:,self.dof_hip_pitch_indices], dim=1)#lower_dof_indices
        return torch.exp(-4 * target_jt_pos_error), target_jt_pos_error
    
    def _reward_freeze_ankle_dof(self):
        # Penalize distance to target joint angles
        target_jt_pos = self.target_jt_pos#+self.default_dof_pos # NOTE(xh): add default pos to target jtdof_hip_pitch_indices
        target_jt_pos_error = torch.mean(torch.abs(self.dof_pos - target_jt_pos)[:,self.dof_ankle_indices], dim=1)#lower_dof_indices
        return torch.exp(-4 * target_jt_pos_error), target_jt_pos_error
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # target_root_local_vel = broadcast_quat_apply(self.target_heading_quat_inv.squeeze(1), 
        #                                     self.target_global_vel[:,:3])
        lin_vel_error = torch.sum(torch.square(self.target_xy_vel - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_vel_sigma), lin_vel_error

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        # target_root_local_ang_vel = broadcast_quat_apply(self.target_heading_quat_inv.squeeze(1), 
        #                                     self.target_global_ang_vel[:,:3])
        ang_vel_error = torch.square(self.target_yaw_vel - self.base_ang_vel[:, 2:]).squeeze(-1)
        return torch.exp(-ang_vel_error / 0.25), ang_vel_error
    
    def _reward_local_pos_phc(self):
        local_body_pos = broadcast_quat_apply(self.heading_quat_inv, 
                                              self.body_pos_raw - 
                                              self.root_states[:, None, :3]).reshape(-1, 3 * self.num_bodies)
        target_local_body_pos = broadcast_quat_apply(self.target_heading_quat_inv, 
                                                     (self.target_global_pos.reshape(-1,self.num_bodies,3) - 
                                                      self.target_global_pos[:,None,:3])).reshape(-1, 3 * self.num_bodies)
        diff_local_body_pos = torch.mean(torch.square(target_local_body_pos - local_body_pos), dim=1)
        return torch.exp(-100 * diff_local_body_pos), diff_local_body_pos
    
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.gait_planner.desired_contact_states

        shaped_force_err = (1 - desired_contact) * (1 - torch.exp(-1 * foot_forces ** 2 / 50.))
        mean_error = shaped_force_err.mean(dim=1)

        stop_mask = self.get_command_stop_mask()
        mean_error[stop_mask] = 0.

        return torch.exp(-mean_error)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.world_foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.gait_planner.desired_contact_states
        shaped_vel_err = desired_contact * (1 - torch.exp(-1 * foot_velocities ** 2 / 2.0))
        mean_error = shaped_vel_err.mean(dim=1)

        stop_mask = self.get_command_stop_mask()
        mean_error[stop_mask] = 0.

        return torch.exp(-mean_error)
    
    def _reward_mismatch_vel(self): # not jump z and fall down rp
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)
        c_update = (lin_mismatch + ang_mismatch) / 2.
        return c_update, c_update
    
    def _reward_tracking_left_hand(self):
        # Tracking the left hand pos in root frame.
        left_hand_pos = self.rigid_body_state[:, self.hand_indices[0], :3]
        local_left_hand_pos = broadcast_quat_apply(quat_inv(self.base_quat), left_hand_pos - self.base_pos)
        error = torch.sum(torch.square(local_left_hand_pos - self.left_hand_touch_commands), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_hand_sigma)
    
    def _reward_tracking_right_hand(self):
        # Tracking the right hand pos in root frame.
        right_hand_pos = self.rigid_body_state[:, self.hand_indices[1], :3]
        local_right_hand_pos = broadcast_quat_apply(quat_inv(self.base_quat), right_hand_pos - self.base_pos)
        error = torch.sum(torch.square(local_right_hand_pos - self.right_hand_touch_commands), dim=1)
        return torch.exp(-error/ self.cfg.rewards.tracking_hand_sigma)

    def _reward_tracking_squat_z(self):
        # Tracking the z of right foot in root coord.
        z_error = torch.square(self.squat_commands[:, 0] - self.rigid_body_state[:, 0, 2])
        return torch.exp(-z_error/ self.cfg.rewards.tracking_squat_sigma)
    
    def _reward_tracking_squat_z_1(self):
        # Tracking the z of right foot in root coord.
        z_error = torch.square(self.squat_commands[:, 0] - self.rigid_body_state[:, 0, 2])
        return torch.exp(-z_error / self.cfg.rewards.tracking_sigma) * (z_error < 0.04)
    
    def _reward_tracking_squat_z_2(self):
        # Tracking the z of right foot in root coord.
        z_error = torch.square(self.squat_commands[:, 0] - self.rigid_body_state[:, 0, 2])
        return torch.exp(-8 * z_error)
    
    def _reward_tracking_squat_z_3(self):
        # Tracking the z of right foot in root coord.
        z_error = torch.square(self.squat_commands[:, 0] - self.rigid_body_state[:, 0, 2])
        return torch.exp(-z_error)
    
    def _reward_tracking_squat_z_larger(self):
        # Tracking the z of right foot in root coord.
        z_error = torch.square(self.squat_commands[:, 0] - self.rigid_body_state[:, 0, 2])
        return torch.exp(-z_error)
    
    def _reward_tracking_squat_z_encourage_difficult(self):
        # Tracking the z of right foot in root coord.
        z_error = torch.square(self.squat_commands[:, 0] - self.rigid_body_state[:, 0, 2])
        z_upper_limit = self.command_ranges["squat_z"][0]
        z_lower_limit = self.command_ranges["squat_z"][1]
        z_middle = (z_upper_limit + z_lower_limit)/2
        z_radius = (z_upper_limit - z_lower_limit)/2
        import ipdb
        ipdb.set_trace()
        if_reach = z_error < 0.0025
        if_middle = (torch.abs(self.squat_commands[:, 0] - z_middle) / z_radius) > 0.6
        if_hard = (torch.abs(self.squat_commands[:, 0] - z_middle) / z_radius) > 0.8
        middle_and_reach = any(reach and middle for reach, middle in zip(if_reach, if_middle))
        rew = torch.exp(-z_error / self.cfg.rewards.tracking_sigma)
        rew[(torch.abs(self.squat_commands[:, 0] - z_middle) / z_radius) > 0.6 and (torch.abs(self.squat_commands[:, 0] - z_middle) / z_radius) < 0.8 and z_error < 0.0025] += 1
        rew[(torch.abs(self.squat_commands[:, 0] - z_middle) / z_radius) > 0.8 and z_error < 0.0025] += 2
        return rew
    
    def _reward_tracking_pitch(self):
        # Tracking the root pitch.
        pitch_error = torch.square(self.pitch_commands[:, 0] - self.rpy[:, 1])
        return torch.exp(-pitch_error)

    def _reward_orientation(self):
        # Penalize non-flat base orientation

        # target_projected_gravity = quat_rotate_inverse(
        #     vec6d_to_quat(self.target_global_ori[:,:6].reshape(-1,3,2)), self.gravity_vec)
        # xy_gravity = torch.sum(torch.square(self.projected_gravity-target_projected_gravity), dim=1)
        # xy_gravity = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        xy_gravity = torch.square(self.projected_gravity[:, 1]) + torch.square(torch.clamp_min(0.13-self.projected_gravity[:, 0],0.))
        return xy_gravity, xy_gravity

    def _reward_energy(self):
        energy = torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
        return energy, energy

    # term 2
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

    def _reward_contact_forces(self):
        contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        waste_force = contact_forces - self.cfg.rewards.max_contact_force
        rew = torch.sum(torch.clip(waste_force, 0., 500), dim=1)  # exceed 500
        return rew, rew

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        collision = torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
        return collision, collision
    
    def _reward_root_roll(self):
        return torch.square(self.rpy[:, 0])
    
    def _reward_root_negative_pitch(self):
        return torch.square(self.rpy[:, 1] * (self.rpy[:, 1] < 0))

    # term 3
    def _reward_action_rate(self):
        # Penalize changes in actions
        action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate, action_rate

    def _reward_hip_yaw_dof_error(self):
        # Penalize arm dof error
        hip_diff = self.dof_pos[:, self.dof_hip_yaw_indices] - self.default_dof_pos[:, self.dof_hip_yaw_indices]
        return torch.sum(torch.square(hip_diff), dim=1), torch.sum(torch.square(hip_diff), dim=1)

    def _reward_hip_roll_dof_error(self):
        # Penalize arm dof error
        hip_diff = self.dof_pos[:, self.dof_hip_roll_indices] - self.default_dof_pos[:, self.dof_hip_roll_indices]
        return torch.sum(torch.square(hip_diff), dim=1), torch.sum(torch.square(hip_diff), dim=1)

    def _reward_ankle_dof_error(self):
        # Penalize arm dof error
        ankle_diff = self.dof_pos[:, self.dof_ankle_indices] - self.default_dof_pos[:, self.dof_ankle_indices]
        return torch.sum(torch.square(ankle_diff), dim=1), torch.sum(torch.square(ankle_diff), dim=1)

    def _reward_feet_away(self):
        # Penalize feet close
        # feet_threshold = 0.4
        # feet_0_pos = self.rigid_body_state[:, self.feet_indices[0], :3]
        # feet_1_pos = self.rigid_body_state[:, self.feet_indices[1], :3]
        # feet_distance = torch.norm(feet_0_pos - feet_1_pos, dim=1)
        # close_penalty = torch.abs(feet_distance - feet_threshold) * (feet_distance < feet_threshold)
        # return close_penalty
        feet_threshold = 0.4
        feet_0_pos = self.rigid_body_state[:, self.feet_indices[0], :3]
        feet_1_pos = self.rigid_body_state[:, self.feet_indices[1], :3]
        feet_distance = torch.norm(feet_0_pos - feet_1_pos, dim=1)
        mask = feet_distance > feet_threshold
        rew = mask * feet_threshold + ~mask * feet_distance
        return rew, feet_distance
    
    def _reward_feet_close(self):
        # Penalize feet away
        feet_threshold = 0.6
        feet_0_pos = self.rigid_body_state[:, self.feet_indices[0], :3]
        feet_1_pos = self.rigid_body_state[:, self.feet_indices[1], :3]
        feet_distance = torch.norm(feet_0_pos - feet_1_pos, dim=1)
        away_penalty = torch.abs(feet_distance - feet_threshold) * (feet_distance > feet_threshold)
        return away_penalty, feet_distance
    
    def _reward_leap_forward(self):
        # Penalize feet away
        # leap_forward_threshold = np.deg2rad(20)
        # leap_penalty = torch.square(self.rpy[:, 1] - leap_forward_threshold) * (self.rpy[:, 1] < leap_forward_threshold)
        # return leap_penalty
        leap_reward = self.rpy[:, 1]
        return leap_reward

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * self.get_command_stop_mask()
        stance_mask = self.get_command_stop_mask() # vel too small
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.  # contact on z axis
        all_contact = torch.all(contact, dim=1)
        non_contact_rew = stance_mask * ~all_contact * 5.
        # contact_rew = stance_mask * all_contact * torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        rew = non_contact_rew #+ contact_rew NOTE
        return rew, rew
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_touch_ground(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        any_contact = torch.any(contact, dim=1).to(torch.float32)  # contact on z axis
        return any_contact

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
    
    def _reward_freeze_upper_body(self):
        # Penalize upper body motion
        # return torch.sum(torch.abs(self.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:, self.upper_dof_indices]), dim=1)
        target_jt_pos_error = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos)[:,self.upper_dof_indices], dim=1)#lower_dof_indices
        return torch.exp(-4 * target_jt_pos_error), target_jt_pos_error

    
    def _reward_freeze_lower_body(self):
        # Penalize upper body motion
        # lower_body_error = torch.sum(torch.abs(self.dof_pos[:, self.lower_dof_indices] - self.default_dof_pos[:, self.lower_dof_indices]), dim=1)
        # return lower_body_error, lower_body_error
        # Penalize distance to target joint angles
        # target_jt_pos = self.target_jt_pos#+self.default_dof_pos # NOTE(xh): add default pos to target jtdof_hip_pitch_indices
        target_jt_pos_error = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos)[:,self.lower_dof_indices], dim=1)#lower_dof_indices
        return torch.exp(-4 * target_jt_pos_error), target_jt_pos_error

    
    def _reward_rand_still(self):
        # Penalize upper body motion
        # return torch.sum(torch.abs(self.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:, self.upper_dof_indices]), dim=1)
        target_jt_pos_error = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * self.rand_still#lower_dof_indices
        return torch.exp(-4 * target_jt_pos_error), target_jt_pos_error
    
    def _reward_arm_dof_error(self):
        arm_err = self.dof_pos[:, self.dof_arm_indices] - self.default_dof_pos[:, self.dof_arm_indices]
        rew = torch.sum(torch.square(arm_err), dim=1)
        return rew, rew

    def _reward_waist_dof_error(self):
        waist_err = self.dof_pos[:, self.dof_waist_indices] - self.default_dof_pos[:, self.dof_waist_indices]
        rew = torch.sum(torch.square(waist_err), dim=1)
        return rew, rew
    
    def _reward_arm_waist_ankle_hip_smoothness(self):
        # arm
        arm_err = torch.mean(
            torch.square(self.actions[:, self.dof_arm_indices] - self.last_actions[:, self.dof_arm_indices]), dim=1)

        # waist
        waist_err = torch.mean(
            torch.square(self.actions[:, self.dof_waist_indices] - self.last_actions[:, self.dof_waist_indices]), dim=1)

        # ankle
        ankle_err = torch.mean(
            torch.square(self.actions[:, self.dof_ankle_indices] - self.last_actions[:, self.dof_ankle_indices]), dim=1)

        # hip
        hip_err = torch.mean(
            torch.square(self.actions[:, self.dof_hip_indices] - self.last_actions[:, self.dof_hip_indices]), dim=1)

        rew = 5 * arm_err + 5 * waist_err + 5 * ankle_err + 5 * hip_err
        return rew, rew
 
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
        
    def _reward_root_ori(self):
        # Tracking of linear velocity commands (xy axes)
        
        target_root_ori = self.target_global_ori[:,:6]
        root_ori_error = torch.sum(torch.square(target_root_ori - quat_to_vec6d(self.root_states[:, 3:7]).reshape(-1,6)), dim=1)
        return torch.exp(-root_ori_error/self.cfg.rewards.tracking_sigma), root_ori_error
    
    def _reward_root_pos(self):
        # Tracking of linear velocity commands (xy axes)
        target_root_pos = self.target_global_pos[:,:3]+ self.env_origins
        root_pos_error = torch.sum(torch.square(target_root_pos - self.root_states[:, :3]), dim=1)
        return torch.exp(-root_pos_error/self.cfg.rewards.tracking_sigma), root_pos_error
    

    def _reward_feet_air_time(self):
        # Reward long steps
        
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # contact_filt = torch.logical_or(contact, self.last_contacts) 
        # self.last_contacts = contact
        # first_contact = (self.feet_air_time > 0.) * contact_filt
        # self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.target_xy_vel, dim=1) > 0.1 #no reward for low ref motion velocity (root xy velocity)
        # self.feet_air_time *= ~contact_filt

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        down = torch.logical_and(contact, ~self.last_contacts) 
        up = torch.logical_and(~contact, self.last_contacts) 
        self.last_contacts = contact
        self.feet_air_time *= ~up
        # first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += (self.dt * ~contact) * torch.any(contact, dim=-1,keepdim=True)
        rew_airTime = torch.min((self.feet_air_time - 0.25), dim=1)[0] * torch.any(down, dim=-1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.target_xy_vel, dim=1) > 0.1 #no reward for low ref motion velocity (root xy velocity)
        rew_airTime *= self.episode_length_buf > 10. #no reward at episode beginning
        return rew_airTime, self.feet_air_time