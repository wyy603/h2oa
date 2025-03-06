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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils.diff_quat import quat_to_vec6d
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import joblib


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, load_run=args.load_run)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.num_envs = 1000
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.test = True
    env_cfg.recycle_data = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.run_name = 'play'

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    print(args.load_run)
    tracking_path = env.cfg.human.filename
    print(tracking_path)
    print('termination_distance', env.cfg.termination.termination_distance)
    
    retarget_data_ori = joblib.load(tracking_path)
    # succ_names = []
    # succ_names = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2020_0.32_names.pkl')

    names_50 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2165_0.5_names.pkl')
    names_32 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2020_0.32_names.pkl')
    names_25 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1949_0.25_names.pkl')
    names_21 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1803_0.21_names.pkl')
    names_18 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1474_0.18_names.pkl')
    names_15 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2036_0.15_names.pkl')
    retarget_data={}
    for name, data in retarget_data_ori.items():
        # if name in names_50:
        if name in names_25:# and name not in names_32:
            retarget_data[name] = data
    # retarget_data=retarget_data_ori
    names_50 = [name for name in names_50 if name not in names_32]
    names_32 = [name for name in names_32 if name not in names_25]
    names_25 = [name for name in names_25 if name not in names_21]
    names_21 = [name for name in names_21 if name not in names_18]
    names_18 = [name for name in names_18 if name not in names_15]

    names_split = [names_50, names_32, names_25, names_21, names_18, names_15]
    names_map = {name: i 
                    for i, names in enumerate(names_split)
                        for name in names}


    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    num_target_seq = env.num_target_seq
    target_seq_len = env.target_seq_len
    max_length = target_seq_len.max()
    # print(num_target_seq)
    succ_buf = torch.zeros(num_target_seq, dtype=torch.float)
    dead_buf = torch.zeros(num_target_seq, dtype=torch.int)

    recycle_data = {}
    # obs_global_buf = [[]*env_cfg.env.num_envs] #torch.tensor([[]*env_cfg.env.num_envs])
    # obs_jt_buf = [[]*env_cfg.env.num_envs] #torch.tensor([[]*env_cfg.env.num_envs])
    obs_global_buf = torch.zeros((env_cfg.env.num_envs, max_length, env.num_bodies*(3+6+3+3)), dtype=torch.float)
    obs_jt_buf = torch.zeros((env_cfg.env.num_envs, max_length, env.num_dofs* 2), dtype=torch.float)
    torque_buf = torch.zeros((env_cfg.env.num_envs, max_length, env.num_dofs), dtype=torch.float)

    # motion_id = 0
    from tqdm import tqdm
    total=num_target_seq*2
    bar = tqdm(total)
    last_n = 0
    obs = env.get_observations()
    all_torque=[0]*6
    error_jt=[0]*6
    error_global=[0]*6
    succ_num = [0]*6
    motion_length = [0]*6
    
    total_recycle_data_num = 0
    while True:
        actions = policy(obs.detach())
        target_j = env.target_jt_j.detach().cpu().clone()
        
        # actions[:] = 0.0
        with torch.no_grad():
            obs, _, rews, dones, infos = env.step(actions.detach()) 
            
        obs_global_buf[torch.arange(env_cfg.env.num_envs),target_j] = infos['obs_global_buf'] # NOTE(xh) IMPORTANT BUG!!!!!!!!!!!!, use torch.arange(env_cfg.env.num_envs)
        obs_jt_buf[torch.arange(env_cfg.env.num_envs),target_j] = infos['obs_jt_buf']
        torque_buf[torch.arange(env_cfg.env.num_envs),target_j] = env.torques.detach().cpu()


        succ_motion_id = infos['succ_id'].to('cpu')
        unique_motion_id, unique_indices = succ_motion_id.unique(return_inverse=True)
        # unique_indices = unique_indices.unique()
        unique_indices = torch.arange(succ_motion_id.size(0)).new_empty(unique_motion_id.size(0)).scatter_(0, unique_indices, torch.arange(succ_motion_id.size(0)))
        # unique_motion_id = infos['succ_id'].to('cpu')[unique_indices]
        unique_env_id = env.time_out_buf.nonzero(as_tuple=False).flatten().cpu()[unique_indices]
        not_yet_succ = succ_buf[unique_motion_id] == 0
        save_env_id = unique_env_id[not_yet_succ]
        save_motion_id = unique_motion_id[not_yet_succ]
        for motion_id, env_id in zip(save_motion_id, save_env_id):
            name = list(retarget_data)[motion_id]
            if name in recycle_data.keys():
                breakpoint()
                continue
            split_id = names_map[name]
            # if name in succ_names:
            #     continue
            total_recycle_data_num += 1
            recycle_data[name] = {
                # 'global_A': retarget_data[name]['global'],
                # 'jt_A': retarget_data[name]['jt'],
                'global': obs_global_buf[env_id,:target_seq_len[motion_id]].detach().cpu().clone().numpy(), 
                'jt': obs_jt_buf[env_id,:target_seq_len[motion_id]].detach().cpu().clone().numpy()
                }
            assert retarget_data[name]['jt'].shape == recycle_data[name]['jt'].shape
            assert retarget_data[name]['global'].shape == recycle_data[name]['global'].shape
            assert (target_j[env_id] + 1 == target_seq_len[motion_id]).all()
            assert (env.target_jt_j[env_id] == 0).all()
            all_torque[split_id] += torque_buf[env_id,:target_seq_len[motion_id]].abs().mean().item()
            error_jt[split_id] += abs((retarget_data[name]['jt'] - recycle_data[name]['jt'])[:,:19]).mean().item()
            error_global[split_id] += abs((retarget_data[name]['global'] - recycle_data[name]['global'])[:,:3*20]).mean().item()
            motion_length[split_id] += target_seq_len[motion_id].item()
            succ_num[split_id] += 1
            succ_buf[motion_id] = 1


        dead_buf[infos['dead_id'].to('cpu').unique()] += 1
        # bar.update(infos['test_motion_id'])
        bar.update(int(infos['test_motion_id']) - last_n)
        last_n = int(infos['test_motion_id'])  # 设置当前进度为 x 
        bar.set_description(f'succ_rate = {succ_buf.mean().item()}')
        # bar.set_description(f'succ_rate = {succ_buf.mean().item()}, jt_error = {error_jt/(succ_buf.sum().item()+1e-10)}, global_error = {error_global/(succ_buf.sum().item()+1e-10)}, motion_length = {motion_length/(succ_buf.sum().item()+1e-10)}')
        if infos['test_motion_id'] > total:
            if (dead_buf>1).all():
                break

    print(succ_buf.mean())
    succ_buf_num = int(succ_buf.sum().item())
    assert succ_buf_num == total_recycle_data_num
    save_dir = os.path.dirname(tracking_path)
    save_name = f'tk_{total_recycle_data_num}_{int(env.cfg.termination.termination_distance*100)}_{args.load_run[-15:]}_' + os.path.basename(tracking_path)
    save_path = os.path.join(save_dir, save_name)
    print(f'save_path = {save_path}')
    log_path = os.path.join(save_dir, 'metrics.csv')
    with open(log_path, 'a') as f:
        for i in range(6):
            motion_num = len(names_split[i])
            f.write(f"{succ_num[i]}-{motion_num}, /, /, /, {all_torque[i]/motion_num:.4g}, {succ_num[i]/motion_num:.4g}, {error_jt[i]/motion_num:.4g}, {error_global[i]/motion_num:.4g}, {motion_length[i]/motion_num:.4g}\n")
    # breakpoint()
    joblib.dump(recycle_data, save_path)
    # joblib.dump(list(recycle_data.keys())+succ_names, os.path.join(save_dir, f'tracked_{total_recycle_data_num}_{env.cfg.termination.termination_distance}_names.pkl'))
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args(test=True)
    play(args)
