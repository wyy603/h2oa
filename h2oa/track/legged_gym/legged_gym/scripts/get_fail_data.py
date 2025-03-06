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

from h2oa.utils import *

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, load_run=args.load_run)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.num_envs = 500
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
    print(env.cfg.human.filename)
    
    retarget_data = joblib.load(env.cfg.human.filename)

    target_jt = []
    target_global = []
    target_length = []
    # start_id = []
    # current_id = 0
    # for name, data in retarget_data.items():
    #     one_target_jt = torch.from_numpy(data['jt'])#.to(device)
    #     one_target_global = torch.from_numpy(data['global'])#.to(device)
    #     target_jt.append(one_target_jt)
    #     target_global.append(one_target_global)
    #     target_length.append(one_target_jt.shape[0])
    # target_jt = torch.cat(target_jt, dim=0)
    # target_global = torch.cat(target_global, dim=0)
    # export policy as a jit module (used to run it from C++)
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

    # recycle_data = [None]*num_target_seq
    succ_name = []
    # obs_global_buf = [[]*env_cfg.env.num_envs] #torch.tensor([[]*env_cfg.env.num_envs])
    # obs_jt_buf = [[]*env_cfg.env.num_envs] #torch.tensor([[]*env_cfg.env.num_envs])
    obs_global_buf = torch.zeros((env_cfg.env.num_envs, max_length, (1+env.num_dofs)*(3+6+3+3)), dtype=torch.float)
    obs_jt_buf = torch.zeros((env_cfg.env.num_envs, max_length, env.num_dofs* 2), dtype=torch.float)

    # motion_id = 0
    from tqdm import tqdm
    total=num_target_seq*2
    bar = tqdm(total)
    last_n = 0
    obs = env.get_observations()
    error_jt=0
    error_global=0
    motion_length = 0
    total_recycle_data_num = 0
    while True:
        actions = policy(obs.detach())
        target_j = env.target_jt_j.detach().cpu().clone()
        target_jt_pos = env.target_jt_pos.detach().cpu().clone()
        # actions[:] = 0.0
        with torch.no_grad():
            obs, _, rews, dones, infos = env.step(actions.detach()) 
        # obs_global_buf = torch.cat((obs_global_buf, infos['recycle_data']['global']), dim=1)
        # obs_jt_buf = torch.cat((obs_jt_buf, infos['recycle_data']['jt']), dim=1)
        # global_body_pos = (env.body_pos_raw - env.env_origins[:,None]).clone().cpu().reshape(env_cfg.env.num_envs,-1)
        # global_body_ori = quat_to_vec6d(env.body_ori_raw).clone().cpu().reshape(env_cfg.env.num_envs,-1)
        # global_body_vel = env.body_vel_raw.clone().cpu().reshape(env_cfg.env.num_envs,-1)
        # global_body_ang_vel = env.body_ang_vel_raw.clone().cpu().reshape(env_cfg.env.num_envs,-1)
        # obs_global_pos = env.target_global_pos.detach().cpu().clone().reshape(env.num_envs,-1)
        # obs_global_pos = (env.body_pos_raw - env.env_origins[:,None]).detach().cpu().clone().reshape(env.num_envs,-1)
        # obs_global_ori = quat_to_vec6d(env.body_ori_raw).detach().cpu().clone().reshape(env.num_envs,-1)
        # obs_global_vel = env.body_vel_raw.detach().cpu().clone().reshape(env.num_envs,-1)
        # obs_global_ang_vel = env.body_ang_vel_raw.detach().cpu().clone().reshape(env.num_envs,-1)
        # env.extras['obs_global'] = torch.cat([obs_global_pos, obs_global_ori, obs_global_vel, obs_global_ang_vel], dim=1)
        # env.extras['obs_jt'] = torch.cat([env.dof_pos,env.dof_vel], dim=1).clone().cpu()
        # env.extras['target_debug'] = env.target_global_pos.clone().cpu()
        # obs_global_buf[torch.arange(env_cfg.env.num_envs),target_j] = infos['obs_global_buf'] # NOTE(xh) IMPORTANT BUG!!!!!!!!!!!!
        # obs_jt_buf[torch.arange(env_cfg.env.num_envs),target_j] = infos['obs_jt_buf']


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
            if name in succ_name:
                breakpoint()
                continue
            total_recycle_data_num += 1
            succ_name.append(name)


        succ_buf[save_motion_id] = 1
        dead_buf[infos['dead_id'].to('cpu').unique()] += 1
        # bar.update(infos['test_motion_id'])
        bar.update(int(infos['test_motion_id']) - last_n)
        last_n = int(infos['test_motion_id'])  # 设置当前进度为 x
        bar.set_description(f'succ_rate = {succ_buf.mean().item()}, jt_error = {error_jt/(succ_buf.sum().item()+1e-10)}, global_error = {error_global/(succ_buf.sum().item()+1e-10)}, motion_length = {motion_length/(succ_buf.sum().item()+1e-10)}')
        if infos['test_motion_id'] > total:
            if (dead_buf>1).all():
                break

    print(succ_buf.mean())
    # succ_buf_num = int(succ_buf.sum().item())
    fail_data={}
    for name, data in retarget_data.items():
        if name not in succ_name:
            fail_data[name] = data

    fail_buf_num = len(fail_data.keys())
    breakpoint()
    assert fail_buf_num == (num_target_seq - total_recycle_data_num)
    print(DATASET / f'PHC/fail_{fail_buf_num}.pkl')
    joblib.dump(fail_data, DATASET / f'PHC/fail_{fail_buf_num}.pkl')
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
