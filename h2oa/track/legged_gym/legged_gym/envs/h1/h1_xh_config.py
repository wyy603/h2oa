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

from legged_gym.envs.base.base_config import BaseConfig

class H1RoughCfg( BaseConfig ):
    class human:
        delay = 0.0 # delay in seconds
        freq = 30
        resample_on_env_reset = True
        filename = '0-Eyes_Japan_Dataset_kanno_walk-13-baggage_on_the_shoulder-kanno_stageii.npy'
        load_global = True
        
    class env:
        num_envs = 2048
        num_dofs = 19
        num_observations = 65 + num_dofs  # TODO
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 19
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        policy_name = "ActorCriticTransformer"

        action_delay = 1  # -1 for no delay
        obs_context_len = 8
        class obs:
            dof_pos = True # dof pos
            dof_vel = True # dof vel

            body_pos = False # body pos in root frame
            body_ori = False # body ori in root frame
            body_vel = False # body vel in root frame
            body_ang_vel = False # body ang_vel in root frame

            root_pos = True # global root pos 
            root_ori = True # global root ori
            root_vel = True # global root vel
            root_ang_vel = True # global root ang_vel

            last_action = True # last policy action

            target_dof_pos = True # target dof pos
            target_dof_vel = False # target dof vel

            target_body_pos = False # target body pos in root frame
            target_body_ori = False # target body ori in root frame
            target_body_vel = False # target body vel in root frame
            target_body_ang_vel = False # target body ang_vel in root frame,

            target_global_pos = False # target global pos in root frame
            target_global_ori = False # target global ori in root frame
            target_global_vel = False # target global vel in root frame
            target_global_ang_vel = False # target global ang_vel in root frame,

            target_root_pos = True # global target root pos, not support for now
            target_root_ori = True # global target root ori, not support for now
            target_root_vel = True # global target root vel, not support for now
            target_root_ang_vel = True # global target root ang_vel, not support for now

            base_orn_rp = False # global base orn_rp
            base_ang_vel = True # base_ang_vel in root frame
            projected_gravity = True # gravity

            commands = False # commands

    class terrain:
        mesh_type = "trimesh" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = True # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.9, 0.9] # min max [m/s]
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

    class init_state:
        pos = [0.0, 0.0, 1.05] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.349,
            'left_knee_joint': 0.698,
            'left_ankle_joint': -0.349,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.349,
            'right_knee_joint': 0.698,
            'right_ankle_joint': -0.349,
            'torso_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.0,
        }

    class control:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'joint': 100.}  # [N*m/rad]
        damping = {'joint': 5.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0
        clip_actions = True
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = 'ankle'
        penalize_contacts_on = []
        terminate_after_contacts_on = ['pelvis', 'hip', 'shoulder', 'elbow', 'knee']
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.0, 2.0]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = -0
            ang_vel_xy = -0
            orientation = -0.
            torques = -0.0
            dof_vel = -0
            dof_acc = -0
            weighted_torques = -0
            base_height = -0. 
            feet_air_time = 0.
            collision = -0.
            feet_stumble = -0.0 
            action_rate = -0
            stand_still = -0.
            dof_pos_limits = -0.0
            target_jt = 0
            target_jt_pos = 1
            target_jt_vel = 0
            energy = -0
            feet_slipping = -0
            
            jt_base_vel_xy_humanplus = 0
            jt_base_ang_vel_yaw_humanplus = 0
            jt_dof_pos_humanplus = 0
            jt_base_rp_humanplus = -0
            energy_humanplus = -0
            jt_foot_contact_humanplus = 0
            feet_slipping_humanplus = -0
            alive_humanplus = 0

            # gp_MaskedMimic = 0.3
            # gr_MaskedMimic = 0.5
            # rh_MaskedMimic = 0.2
            # jv_MaskedMimic = 0.1
            # jav_MaskedMimic = 0.1
            # eg_MaskedMimic = 0.0002
            root_lin_vel = 0.1
            root_ang_vel = 0.1
            root_ori = 0.1
            root_pos = 0.1

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        use_privileged_rew = True

    class termination:
        r_threshold = 0.5
        p_threshold = 0.5
        z_threshold = 0.
        contact_threshold = 1

    class normalization:
        class obs_scales:
            # lin_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05

            body_pos = 1.0
            body_ori = 1.0
            body_vel = 0.05
            body_ang_vel = 0.05

            root_pos = 1.0
            root_ori = 1.0
            root_vel = 0.05
            root_ang_vel = 0.05

            last_action = 1.0

            target_dof_pos = 1.0
            target_dof_vel = 0.05

            target_body_pos = 1.0
            target_body_ori = 1.0
            target_body_vel = 0.05
            target_body_ang_vel = 0.05

            target_global_pos = 1.0
            target_global_ori = 1.0
            target_global_vel = 0.05
            target_global_ang_vel = 0.05

            target_root_pos = 1.0
            target_root_ori = 1.0
            target_root_vel = 0.05
            target_root_ang_vel = 0.05

            base_ang_vel = 0.25
            base_orn_rp = 1.0
            projected_gravity = 1

            height_measurements = 5.0

        commands_scale = [1., 1., 1., 1.]
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            # lin_vel = 0.1
            orn = 0.05
            ang_vel = 0.2
            # gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 2
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class H1RoughCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512, 256, 128]
        # activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 1e-5
        num_learning_epochs = 2
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.5e-4
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCriticTransformer'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32 # per iteration
        max_iterations = 15000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'rough_h1'
        run_name = None
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt