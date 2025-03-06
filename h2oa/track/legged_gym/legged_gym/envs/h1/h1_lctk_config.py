from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1RoughCfg(LeggedRobotCfg):
    class human:
        delay = 0.0 # delay in seconds
        freq = 15
        # filename = '/cephfs_yili/shared/xuehan/H1_RL/dn_11438_25_all__o8_s819000_retarget_13911_amass_train_13912.pkl'
        # filename = '/cephfs_yili/shared/xuehan/H1_RL/dn_8198_h2o.pkl'
        # filename = '/home/ubuntu/data/MDM/dn_35hand_mdm.pkl'
        filename = '/home/ubuntu/data/MDM/dn_10_box_mdm.pkl'
        # filename = '/home/ubuntu/data/PHC/nvel_dn_35_mdm.pkl'
        rand_still=0.2

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.9]  # x,y,z [m]
        # default_joint_angles = {  # = target angles [rad] when action = 0.0
        #     'left_hip_yaw_joint': 0.,
        #     'left_hip_roll_joint': 0,
        #     'left_hip_pitch_joint': -0.4,
        #     'left_knee_joint': 0.8,
        #     'left_ankle_joint': -0.4,
        #     'right_hip_yaw_joint': 0.,
        #     'right_hip_roll_joint': 0,
        #     'right_hip_pitch_joint': -0.4,
        #     'right_knee_joint': 0.8,
        #     'right_ankle_joint': -0.4,
        #     'torso_joint': 0.,
        #     'left_shoulder_pitch_joint': 0.,
        #     'left_shoulder_roll_joint': 0,
        #     'left_shoulder_yaw_joint': 0.,
        #     'left_elbow_joint': 0.,
        #     'right_shoulder_pitch_joint': 0.,
        #     'right_shoulder_roll_joint': 0.0,
        #     'right_shoulder_yaw_joint': 0.,
        #     'right_elbow_joint': 0.,
        # }

        # Unitree V2
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

        default_joint_angles = {
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.45,
            'left_knee_joint': 1.05,
            'left_ankle_joint': -0.585,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.45,
            'right_knee_joint': 1.05,
            'right_ankle_joint': -0.585,
            'torso_joint': 0.0,
            'left_shoulder_pitch_joint': 0.44,
            'left_shoulder_roll_joint': -0.01,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.308,
            'right_shoulder_pitch_joint': 0.44,
            'right_shoulder_roll_joint': 0.01,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.308
        }

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        observe_gait_commands = False
        show_touch_point = False
        plane_terrain = False
        num_privileged_obs = 145 + 40
        num_observations = 145
        num_proprio = 66
        num_actions = 19
        num_torques = 19
        num_bodies = 22
        termination_roll = 0.5
        termination_pitch = 0.5 # 1.5

    class commands(LeggedRobotCfg.commands):
        num_commands = 6  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        loco_command = True  # 3
        heading_command = False  # 1
        hand_touch_command = False  # 3
        squat_command = False  # 1
        pitch_command = False # 1 ***


        # lin_vel_clip = 0.05
        # ang_vel_clip = 0.05
        
        mask = False
        skill_curriculum = False
        skill_curriculum_strategy = 2
        skill_curriculum_interval = 2000
        skill_curriculum_list = ["multi_single_skill", "multi_skill"]
        skill_rew_map = {
            "locomotion" : ["tracking_lin_vel", "tracking_ang_vel"],
            "squat_down" : ["tracking_squat_z"],
            "touch_point" : ["tracking_left_hand", "tracking_right_hand"],
            "multi_single_skill" : ["tracking_lin_vel", "tracking_ang_vel", "tracking_squat_z", "tracking_left_hand", "tracking_right_hand"],
            "multi_skill": ["tracking_lin_vel", "tracking_ang_vel", "tracking_squat_z", "tracking_left_hand", "tracking_right_hand"]
        }
        skill_rew_list = ["tracking_lin_vel", "tracking_ang_vel", "tracking_squat_z", "tracking_left_hand", "tracking_right_hand"]
        skill_curriculum_loop = 1
        command_curriculum = False
        command_curriculum_interval = 500
        command_highway = False
        command_stophighway = 10000
        command_teacher = False
        command_teacher_ratio = 0.1

        class ranges(LeggedRobotCfg.commands.ranges):
            # lin_vel_x = [0., 0.]  # min max [m/s]
            # lin_vel_y = [0., 0.]  # min max [m/s]
            # ang_vel_yaw = [0., 0.]  # min max [rad/s]

            touch_x_left = [0.33, 0.33]
            touch_y_left = [0.215, 0.215]
            touch_z_left = [0.08, 0.08]

            touch_x_right = [0.33, 0.33]
            touch_y_right = [-0.215, -0.215]
            touch_z_right = [0.08, 0.08]
            
            squat_z = [0.98, 0.98]

            pitch_forward = [0.0, 0.8]
            # heading = [-3.14, 3.14]0.33, 0.215, 0.08

            lin_vel_x = [.0, 0.]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]

            # touch_x_left = [-0.4, 0.6]
            # touch_y_left = [0.1, 0.8]
            # touch_z_left = [-0.3, 0.6]

            # touch_x_right = [-0.4, 0.6]
            # touch_y_right = [-0.1, -0.8]
            # touch_z_right = [-0.3, 0.6]

        class initial_curriculum_ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0., 0.]  # min max [m/s]
            lin_vel_y = [-0., 0.]  # min max [m/s]
            ang_vel_yaw = [-0., 0.]  # min max [rad/s]

            touch_x_left = [0.33, 0.33]
            touch_y_left = [0.215, 0.215]
            touch_z_left = [0.08, 0.08]

            touch_x_right = [0.33, 0.33]
            touch_y_right = [-0.215, -0.215]
            touch_z_right = [0.08, 0.08]
            
            squat_z = [0.8, 1.0]
            # heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        # todo: add waist (torso) joint
        stiffness = {
            # legs
            'hip_yaw': 60,
            'hip_roll': 220,
            'hip_pitch': 220,
            'knee': 320,
            'ankle': 40,
            # body
            'torso': 200,
            # arms
            'shoulder_pitch': 40,
            'shoulder_roll': 40,
            'shoulder_yaw': 18,
            "elbow": 20,
        }  # [N*m/rad]
        damping = {
            # legs
            'hip_yaw': 1.5,
            'hip_roll': 4.,
            'hip_pitch': 4.,
            'knee': 4.,
            'ankle': 2.,
            # body
            'torso': 3.,
            # arms
            'shoulder_pitch': 1.0,
            'shoulder_roll': 1.0,
            'shoulder_yaw': 0.5,
            "elbow": 0.5,
        }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        # class gait:
        #     gait_command = True
        #     frequencies = 1.5
        #     phase_offset = 0.5
        #     stance_ratio = 0.7
        #     kappa_gait_probs = 0.07

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_add_hand_link.urdf'
        name = "h1"
        foot_name = "ankle"
        hand_name = "hand"
        penalize_contacts_on = ['hip', 'knee', 'torso']
        terminate_after_contacts_on = ['pelvis']
        upper_dof_names = ["torso", "shoulder", "elbow"]
        upper_dof_names_no_left = ["torso", "right_shoulder", "right_elbow"]
        upper_dof_names_no_right = ["torso", "left_shoulder", "left_elbow"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
        force_sensor_configs = {
            # 'left_foot_front': {'link_name': 'left_ankle_link', 'position': (0.14, 0.0, -0.09)},
            # 'left_foot_rear': {'link_name': 'left_ankle_link', 'position': (-0.07, 0.0, -0.09)},
            # 'right_foot_front': {'link_name': 'right_ankle_link', 'position': (0.14, 0.0, -0.09)},
            # 'right_foot_rear': {'link_name': 'right_ankle_link', 'position': (-0.07, 0.0, -0.09)},
            'left_foot': {'link_name': 'left_ankle_link', 'position': (0.0, 0.0, -0.07)},
            'right_foot': {'link_name': 'right_ankle_link', 'position': (0.0, 0.0, -0.07)},
        }

    class domain_rand(LeggedRobotCfg.domain_rand):
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

        randomize_proprio_latency = True
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


    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.98
        max_contact_force = 500.
        # default 1, 0.125, 0.25
        tracking_vel_sigma = 1
        tracking_squat_sigma = 0.125
        tracking_hand_sigma = 0.25


        class scales: # TODO
            # task 
            tracking_lin_vel = 2.0
            tracking_ang_vel = 4.0
            mismatch_vel = 2.0 # NOTE
            target_jt_pos = 0
            local_pos_phc = 1 # only for metrics
            target_updof = 5
            target_lwdof = 0
            freeze_lower_body = 0
            freeze_upper_body = 0
            rand_still = 5
            
            feet_air_time = 0

            # tracking_squat_z = 1.0
            # tracking_left_hand = 1.0
            # tracking_right_hand = 1.0
            # tracking_pitch = 1.0
            # tracking_contacts_shaped_force = 1
            # tracking_contacts_shaped_vel = 1

            # behavior constraints
            stand_still = -1.0 # NOTE
            orientation = -5. # NOTE target_projected_gravity error
            collision = -5.
            hip_yaw_dof_error = -5. # NOTE
            hip_roll_dof_error = -1
            ankle_dof_error = -1
            feet_away = .0
            feet_close = -0.
            # lin_vel_z = -0.5
            # feet_close = -0.5
            # freeze_upper_body = -0.5
            # leap_forward = 5
            # root_roll = -5.
            # root_negative_pitch = -5.

            arm_dof_error = -.0 # NOTE
            waist_dof_error = -1.0 # NOTE
            arm_waist_ankle_hip_smoothness = -0.03

            # energy
            dof_vel = -1.e-4
            dof_acc = -2.e-6
            energy = -3e-7
            action_rate = -1e-4
            weighted_torques = -1.e-7
            contact_forces = -2.e-4

            # added 0716
            dof_pos_limits = -1.0
            action_smoothness = -0.01

        # class scales:
        #     # task
        #     tracking_lin_vel = 2.0
        #     tracking_ang_vel = 1.5
        #     mismatch_vel = 1.0
        #     base_height = -10.0
        #     orientation = -5.0

        #     # behavior constraints
        #     stand_still = -1.0
        #     touch_ground = 0.1
        #     dof_pos_limits = -10.0
        #     collision = -10.0
        #     arm_dof_error = -1.0
        #     waist_dof_error = -5.0
        #     hip_yaw_dof_error = -0.3
        #     feet_away = 0.6
        #     arm_waist_ankle_smoothness = -0.01
        #     action_smoothness = -0.002

        #     # energy
        #     dof_vel = -2.e-4
        #     dof_acc = -5.e-7
        #     energy = -2.5e-7
        #     action_rate = -1e-3
        #     weighted_torques = -2.e-5
        #     contact_forces = -3e-4

        #     # banned
        #     # feet_air_time = 2.0
        #     # stand_still = 0.5
        #     # touch_ground = 0.1
        #     # collision = -10.0

    class normalization:
        class obs_scales:
            orn = 0.05
            lin_vel = 2.0
            roll_pitch = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            cmd = 0

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            roll_pitch = 0.02
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            # height_measurements = 0.1

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1  # for fast
        horizontal_scale = 0.05  # [m] influence computation time by a lot
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        height = [0.00, 0.05]  # [0.04, 0.1]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False
        teleport_thresh = 0.3
        teleport_robots = False

        all_vertical = False
        no_flat = True

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 10  # number of terrain cols (types)

        terrain_dict = {"smooth slope": 0.,
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0.,
                        "rough stairs down": 0.,
                        "discrete": 0.0,
                        "stepping stones": 0.,
                        "gaps": 0.,
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = None  # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False


class H1HistoryCfg(H1RoughCfg):
    class env(H1RoughCfg.env):
        num_history = 4
        num_prop = 38
        num_observations = num_prop * (num_history + 1)


class H1RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 42
    runner_class_name = 'OnPolicyRunner'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 5e-3 # TODO
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 30000  # number of policy updates

        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = 'h1-s2r'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt