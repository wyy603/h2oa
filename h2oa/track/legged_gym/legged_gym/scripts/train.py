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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
# args.debug=False

# os.environ["WANDB_API_KEY"] = 'd467077c5e839616092d5f0761606c729cc6108c'
# os.environ["WANDB_MODE"] = "offline"
def train(args):
    if not args.debug:
        import wandb
        wandb.init(
            project='H1-RL', 
            name=args.run_name, 
            entity="axian",
            mode='offline'
            )
        wandb.save(LEGGED_GYM_ENVS_DIR + f"/h1/{args.task}_config.py", policy="now")
        wandb.save(LEGGED_GYM_ENVS_DIR + f"/h1/{args.task}.py", policy="now")
    # wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/modules/actor_critic.py", policy="now")
    # wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/algorithms/ppo.py", policy="now")
    # wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/runners/on_policy_runner.py", policy="now")

        

    env, env_cfg = task_registry.make_env(name=args.task, args=args,debug=args.debug)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=None if args.debug else 'default')
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
