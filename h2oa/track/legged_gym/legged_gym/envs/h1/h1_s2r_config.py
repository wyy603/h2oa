from legged_gym.envs.base.base_config import BaseConfig

class H1RoughCfg( BaseConfig ):
    class human:
        delay = 0.0 # delay in seconds
        freq = 20
        resample_on_env_reset = True
        # filename = '/cephfs_yili/shared/xuehan/H1_RL/dn_11438_25_all__o8_s819000_retarget_13911_amass_train_13912.pkl'
        # filename = '/cephfs_yili/shared/xuehan/H1_RL/dn_8198_h2o.pkl'
        # filename = '/cephfs_yili/shared/xuehan/H1_RL/dn_11_mdm.pkl'
        filename = '/cephfs_yili/shared/xuehan/H1_RL/dn_35_mdm.pkl'
        # filename = '/home/ubuntu/data/PHC/train_500.pkl'
        # filename = '/home/ubuntu/data/PHC/dn_35_mdm.pkl'
        load_global = True
        multi_motion = True
        
    class env:
        num_envs = 8192
        num_dofs = 19
        num_bodies = 22
        num_privileged_obs = 40 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 19
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        policy_name = "ActorCritic"

        action_delay = -1  # -1 for no delay
        obs_context_len = 8
        target_init = False
        target_heading_init = True # init heading should be identity
        z_norm=True
        curriculum_lowerband = 50
        curriculum_upperband = 60
        extend_frames = 20
        class obs:
            ############### self ################
            dof_pos = True # dof pos

            body_pos = False # body pos in heading frame

            ############### target ################
            target_dof_pos = True # target dof pos

            target_body_pos = False # target body pos in target heading frame
            target_global_pos = False # target body pos in target heading frame

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
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # default_joint_angles = {
        #     'left_hip_yaw_joint': 0.0,
        #     'left_hip_roll_joint': 0.0,
        #     'left_hip_pitch_joint': -0.55,
        #     'left_knee_joint': 1.24,
        #     'left_ankle_joint': -0.7,
        #     'right_hip_yaw_joint': 0.0,
        #     'right_hip_roll_joint': 0.0,
        #     'right_hip_pitch_joint': -0.55,
        #     'right_knee_joint': 1.24,
        #     'right_ankle_joint': -0.7,
        #     'torso_joint': 0.0,
        #     'left_shoulder_pitch_joint': 0.4,
        #     'left_shoulder_roll_joint': -0.015,
        #     'left_shoulder_yaw_joint': 0.0,
        #     'left_elbow_joint': 0.31,
        #     'right_shoulder_pitch_joint': 0.4,
        #     'right_shoulder_roll_joint': 0.015,
        #     'right_shoulder_yaw_joint': -0.0,
        #     'right_elbow_joint': 0.31
        # }
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
        random_episode_lenth = -100 # <0 disable 

    class control:
        control_type = 'P'
        # jl setting:
        # stiffness = {
        #     # legs
        #     'hip_yaw': 60,
        #     'hip_roll': 220,
        #     'hip_pitch': 220,
        #     'knee': 320,
        #     'ankle': 40,
        #     # body
        #     'torso': 200,
        #     # arms
        #     'shoulder_pitch': 40,
        #     'shoulder_roll': 40,
        #     'shoulder_yaw': 18,
        #     "elbow": 20,
        # }  # [N*m/rad]
        # damping = {
        #     # legs
        #     'hip_yaw': 1.5,
        #     'hip_roll': 4.,
        #     'hip_pitch': 4.,
        #     'knee': 4.,
        #     'ankle': 2.,
        #     # body
        #     'torso': 3.,
        #     # arms
        #     'shoulder_pitch': 1.0,
        #     'shoulder_roll': 1.0,
        #     'shoulder_yaw': 0.5,
        #     "elbow": 0.5,
        # }
        # action_scale = 0.25

        # xh setting:
        stiffness = {'joint': 100.}  # [N*m/rad]
        damping = {'joint': 5.}     # [N*m*s/rad]
        action_scale = 1.0

        # H2O setting:
        # stiffness = {'hip_yaw': 200,
        #              'hip_roll': 200,
        #              'hip_pitch': 200,
        #              'knee': 300,
        #              'ankle': 40,
        #              'torso': 300,
        #              'shoulder': 100,
        #              "elbow":100,
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 5,
        #              'hip_roll': 5,
        #              'hip_pitch': 5,
        #              'knee': 6,
        #              'ankle': 2,
        #              'torso': 6,
        #              'shoulder': 2,
        #              "elbow":2,
        #              }  # [N*m/rad]  # [N*m*s/rad]
        # action_scale = 0.25


        clip_actions = True
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_add_hand_link.urdf'
        name = "h1"
        foot_name = "ankle"
        hand_name = "hand"
        knee_name = "knee"
        penalize_contacts_on = ['hip', 'knee', 'torso']
        # zzk setting
        # terminate_after_contacts_on = ['pelvis']
        # xh setting
        terminate_after_contacts_on = ['pelvis', 'hip', 'shoulder', 'elbow', 'knee']
        upper_dof_names = ["torso", "shoulder", "elbow"]
        right_arm_dof_names = ["right_shoulder", "right_elbow"]
        left_arm_dof_names = ["left_shoulder", "left_elbow"]
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
        force_sensor_configs = {
            # 'left_foot_front': {'link_name': 'left_ankle_link', 'position': (0.14, 0.0, -0.09)},
            # 'left_foot_rear': {'link_name': 'left_ankle_link', 'position': (-0.07, 0.0, -0.09)},
            # 'right_foot_front': {'link_name': 'right_ankle_link', 'position': (0.14, 0.0, -0.09)},
            # 'right_foot_rear': {'link_name': 'right_ankle_link', 'position': (-0.07, 0.0, -0.09)},
            'left_foot': {'link_name': 'left_ankle_link', 'position': (0.0, 0.0, -0.07)},
            'right_foot': {'link_name': 'right_ankle_link', 'position': (0.0, 0.0, -0.07)},
        }

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.0, 2.0]

        randomize_base_mass = True  #
        added_mass_range = [-1., 5.]

        randomize_base_com = True
        added_com_range_x = [-0.1, 0.1]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.2, 0.2]

        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]
        arm_motor_strength_range = [0.8, 1.2]

        randomize_proprio_latency = False
        proprio_latency_range = [0.005, 0.045]

        push_robots = True
        push_interval_s = 10.
        max_push_vel_xy = 1.

        randomize_gripper_mass = False
        gripper_added_mass_range = [0, 0.1]

        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]

        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]

        reset_arm_dof_when_resampling_command = False
        reset_arm_dof_when_push_robots = False


    class rewards:
        class scales:
            # ############ penalty ############
            collision = -5.
            # punish_large_vel = 0.2
            # punish_sliding = 0.2

            # ########### regularization ###########
            dof_vel = -1.e-4
            dof_acc = -2.e-6
            energy = -3e-7
            action_rate = -1e-4
            weighted_torques = -1.e-7
            contact_forces = -2.e-4
            dof_pos_limits = -1.0
            action_smoothness = -0.002
            

            # ########## task reward #########
            alive = 1

            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            target_jt = 0
            target_jt_pos = 0
            target_jt_vel = 0
            root_lin_vel = 10
            root_ang_vel = 10
            root_ori = 10
            root_pos = 10

            global_pos_phc = 10
            global_ori_phc = 5
            global_vel_phc = 0.
            global_ang_vel_phc = 0.
            local_pos_phc = 10
            local_ori_phc = 5
            local_vel_phc = 0.
            local_ang_vel_phc = 0.


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 10 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        # base_height_target = 1.
        max_contact_force = 500. # forces above this value are penalized
        use_privileged_rew = True
        use_curriculum = True

    class termination:
        r_threshold = 0.5
        p_threshold = 0.5
        z_threshold = 0.5
        contact_threshold = 1
        enable_early_termination = True
        termination_distance = 0.5

    class normalization:
        class obs_scales:
            lin_vel = 2.0
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

            diff_local_pos = 1.0
            diff_local_ori = 1.0
            diff_local_vel = 0.05
            diff_local_ang_vel = 0.05

            diff_global_pos = 1.0
            diff_global_ori = 1.0
            diff_global_vel = 0.05
            diff_global_ang_vel = 0.05

            target_root_pos = 1.0
            target_root_ori = 1.0
            target_root_vel = 0.05
            target_root_ang_vel = 0.05

            base_lin_vel = 0.25
            base_ang_vel = 0.25
            base_orn_rp = 1.0
            projected_gravity = 0.25 # NOTE: 1

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

    class terrain:
        # xh setting
        mesh_type = "plane" # none, plane, heightfield or trimesh
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

        # jl setting
        # mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        # hf2mesh_method = "fast"  # grid or fast
        # max_error = 0.1  # for fast
        # horizontal_scale = 0.05  # [m] influence computation time by a lot
        # vertical_scale = 0.005  # [m]
        # border_size = 25  # [m]
        # height = [0.0, 0.05]  # [m] 0.05, 0.1
        # gap_size = [0.02, 0.1]
        # stepping_stone_distance = [0.02, 0.08]
        # downsampled_scale = 0.075
        # curriculum = False
        # teleport_thresh = 0.3
        # teleport_robots = False

        # all_vertical = False
        # no_flat = True

        # static_friction = 1.0
        # dynamic_friction = 1.0
        # restitution = 0.

        # measure_heights = False
        # # 1mx1m rectangle (without center line)
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        #                      0.8]  # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        # selected = False  # select a unique terrain type and pass all arguments
        # terrain_kwargs = None  # Dict of arguments for selected terrain
        # max_init_terrain_level = 5  # starting curriculum state
        # terrain_length = 8.
        # terrain_width = 8.
        # num_rows = 10  # number of terrain rows (levels)  # spreaded is benifitiall !
        # num_cols = 10  # number of terrain cols (types)

        # terrain_dict = {"smooth slope": 0.,
        #                 "rough slope up": 0.,
        #                 "rough slope down": 0.,
        #                 "rough stairs up": 0.,
        #                 "rough stairs down": 0.,
        #                 "discrete": 0.0,
        #                 "stepping stones": 0.,
        #                 "gaps": 0.,
        #                 "rough flat": 1.0,
        #                 "pit": 0.0,
        #                 "wall": 0.0}
        # terrain_proportions = list(terrain_dict.values())
        # # trimesh only:
        # slope_treshold = None  # slopes above this threshold will be corrected to vertical surfaces
        # origin_zero_z = False

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
        entropy_coef =  1e-4 # IMPORTANT 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1e-3
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        # policy_class_name = 'ActorCriticTransformer'
        # algorithm_class_name = 'PPO'
        # num_steps_per_env = 32 # per iteration
        # max_iterations = 15000 # number of policy updates

        # # logging
        # save_interval = 200 # check for potential saves every this many iterations
        # experiment_name = 'rough_h1'
        # run_name = None
        # # load and resume
        # resume = False
        # load_run = -1 # -1 = last run
        # checkpoint = -1 # -1 = last saved model
        # resume_path = None # updated from load_run and chkpt
        max_iterations = 20000  # number of policy updates

        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = 'h1-s2r'
        run_name = None
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt