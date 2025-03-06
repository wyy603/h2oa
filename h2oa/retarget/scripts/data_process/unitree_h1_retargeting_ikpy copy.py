"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

import math
import numpy as np
import torch
import random
import time
import os
import ipdb
import ikpy.chain
import ikpy.utils.plot as plot_utils
import joblib
import sys
from scipy.spatial.transform import Rotation as sRot
sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
    {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")


# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
debug = True
if not debug:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")


asset_root = "/home/xuehan/software/isaacgym/assets/"

# load h1 asset
h1_asset_file = "h1_description/urdf/h1_with_hand.urdf"
# h1_asset_file = '/home/xuehan/workspace/PHC/phc/data/assets/mjcf/h1.xml'
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
h1_asset = gym.load_asset(sim, asset_root, h1_asset_file, asset_options)

# ipdb.set_trace()
h1_dof_names = gym.get_asset_dof_names(h1_asset)
h1_dof_props = gym.get_asset_dof_properties(h1_asset)
h1_num_dofs = gym.get_asset_dof_count(h1_asset)
h1_dof_states = np.zeros(h1_num_dofs, dtype=gymapi.DofState.dtype)
h1_dof_types = [gym.get_asset_dof_type(h1_asset, i) for i in range(h1_num_dofs)]
h1_dof_positions = h1_dof_states['pos']
h1_lower_limits = h1_dof_props["lower"]
h1_upper_limits = h1_dof_props["upper"]
h1_ranges = h1_upper_limits - h1_lower_limits
h1_mids = 0.3 * (h1_upper_limits + h1_lower_limits)
h1_stiffnesses = h1_dof_props['stiffness']
h1_dampings = h1_dof_props['damping']
h1_armatures = h1_dof_props['armature']
h1_has_limits = h1_dof_props['hasLimits']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(h1_num_dofs)
speeds = np.zeros(h1_num_dofs)
for i in range(h1_num_dofs):
    if h1_has_limits[i]:
        if h1_dof_types[i] == gymapi.DOF_ROTATION:
            h1_lower_limits[i] = clamp(h1_lower_limits[i], -math.pi, math.pi)
            h1_upper_limits[i] = clamp(h1_upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if h1_lower_limits[i] > 0.0:
            defaults[i] = h1_lower_limits[i]
        elif h1_upper_limits[i] < 0.0:
            defaults[i] = h1_upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if h1_dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            h1_lower_limits[i] = -math.pi
            h1_upper_limits[i] = math.pi
        elif h1_dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            h1_lower_limits[i] = -1.0
            h1_upper_limits[i] = 1.0
    # set DOF position to default
    h1_dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    # ipdb.set_trace()
    if h1_dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (h1_upper_limits[i] - h1_lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (h1_upper_limits[i] - h1_lower_limits[i]), 0.1, 7.0)

# load human motion data
motion_root = "/home/xuehan/workspace/humanplus/HST/isaacgym/human_motion_data/controlvae_human_joint_pos_data"
# motion_root = "../../../data/joint_pos_data"
motion_name = "walk.npy"
motion_path = os.path.join(motion_root, motion_name)
motion_data = np.load(motion_path)
# in_file='/home/xuehan/workspace/PHC/sample_data/amass_copycat_take6_train.pkl'
skeleton_tree = SkeletonTree.from_mjcf("/home/xuehan/workspace/PHC/phc/data/assets/mjcf/smpl_humanoid_1.xml")
in_file='/home/xuehan/workspace/PHC/sample_data/amass_copycat_take6_test.pkl' 
amass_data = joblib.load(in_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
amass_data = amass_data['0-Transitions_mazen_c3d_jumpingjacks_longjump_stageii']
pose_aa=amass_data['pose_aa'].reshape(-1,24,3)
pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(-1, 24, 4)
new_sk_state = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
    torch.from_numpy(pose_quat),
    torch.zeros_like(torch.from_numpy(amass_data['trans'])), # move motion data to origin
    is_local=True)
local_motion_data = new_sk_state.global_translation
# %matplotlib widget
# import matplotlib.pyplot as plt
# fig, ax = plot_utils.init_3d_figure()
# my_chain.plot(my_chain.inverse_kinematics(target_position), ax, target=target_position)
# plt.xlim(-0.1, 0.1)
# plt.ylim(-0.1, 0.1)
# motion_data['trans']
# motion_data['pose_6d']

# motion_data -= [motion_data[0,0,0], 0.0, motion_data[0,0,2]] # move motion data to origin
# motion_data_x = motion_data[:, :, :1]
# motion_data_y = motion_data[:, :, 1:2]
# motion_data_z = motion_data[:, :, 2:]
# motion_data = np.concatenate([motion_data_z, -motion_data_x, motion_data_y], axis = -1)
# local_motion_data = motion_data - motion_data[:, :1, :]
# human_joint_names = ['pelvis', 'pelvis_lowerback', 'lowerback_torso', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_toes', 'right_toes', 'neck', 'left_torso_clavicle', 'right_torso_clavicle', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
human_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

h1_dof_names = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joright_hand_jointint', 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_hand_joint', 'L_index_proximal_joint', 'L_index_intermediate_joint', 'L_middle_proximal_joint', 'L_middle_intermediate_joint', 'L_pinky_proximal_joint', 'L_pinky_intermediate_joint', 'L_ring_proximal_joint', 'L_ring_intermediate_joint', 'L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'L_thumb_distal_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_hand_joint', 'R_index_proximal_joint', 'R_index_intermediate_joint', 'R_middle_proximal_joint', 'R_middle_intermediate_joint', 'R_pinky_proximal_joint', 'R_pinky_intermediate_joint', 'R_ring_proximal_joint', 'R_ring_intermediate_joint', 'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_intermediate_joint', 'R_thumb_distal_joint']
# joint list for retargeting
true_h1_joint_names = ["torso_joint", "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint", "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_hand_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_hand_joint"]
true_h1_joint_parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 0, 16, 17, 18, 19]
# correspondence joint dict between h1 and human
# correspondence_dict = {"left_ankle_joint": "left_ankle", "right_ankle_joint" : "right_ankle", "left_knee_joint": "left_knee", "right_knee_joint": "right_knee", "left_elbow_joint": "left_elbow", "right_elbow_joint": "right_elbow", "left_hand_joint": "left_wrist", "right_hand_joint": "right_wrist"}
# ipdb.set_trace()
# configure env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

# position the camera
cam_pos = gymapi.Vec3(5.2, 2.0, 5)
cam_target = gymapi.Vec3(5, -2.5, 13)

envs = []
actor_handles = []
joint_handles = {}
h1_joint_pos_list = []
h1_joint_ori_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.08, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    
    # get h1 joint pos, ori
    if i == 0:
        joint_pose = gym.get_actor_joint_transforms(env, actor_handle)
        joint_names = gym.get_actor_joint_names(env, actor_handle)
        for j in range(len(true_h1_joint_names)):
            cur_joint_pose = joint_pose[joint_names.index(true_h1_joint_names[j])] 
            h1_joint_pos_list.append(np.array([[cur_joint_pose[0][0], cur_joint_pose[0][1], cur_joint_pose[0][2]]]))
            h1_joint_ori_list.append(np.array([[cur_joint_pose[1][0], cur_joint_pose[1][1], cur_joint_pose[1][2], cur_joint_pose[1][3]]]))         

# set pos, ori, rot, offset
# ipdb.set_trace()
# true_h1_joint_position = np.concatenate(h1_joint_pos_list, axis=0)
# true_h1_joint_orientation = np.zeros([len(true_h1_joint_names), 3])
# true_h1_joint_rotation = np.zeros([len(true_h1_joint_names), 3])
# true_h1_joint_offset = [true_h1_joint_position[i] - true_h1_joint_position[true_h1_joint_parents[i]] for i in
#                     range(len(true_h1_joint_position))]
# true_h1_joint_offset[0] = np.array([0.0, 0.0, 0.0])

# define kinematic chains
left_leg_chain = ikpy.chain.Chain.from_urdf_file(asset_root+"h1_description/urdf/h1_with_hand.urdf", base_elements=["pelvis", "left_hip_yaw_joint"], name="left leg" )
right_leg_chain = ikpy.chain.Chain.from_urdf_file(asset_root+"h1_description/urdf/h1_with_hand.urdf", base_elements=["pelvis", "right_hip_yaw_joint"], name="right leg" )
left_arm_chain = ikpy.chain.Chain.from_urdf_file(asset_root+"h1_description/urdf/h1_with_hand.urdf", base_elements=["pelvis", "left_shoulder_pitch_joint"], name="left arm" )
right_arm_chain = ikpy.chain.Chain.from_urdf_file(asset_root+"h1_description/urdf/h1_with_hand.urdf", base_elements=["pelvis", "right_shoulder_pitch_joint"], name="right arm",n_links=5)
# forward kinematics
from IPython import embed;embed()
h1_left_ankle_pos = left_leg_chain.forward_kinematics([0, 0, 0, 0, 0, 0])
h1_right_ankle_pos = right_leg_chain.forward_kinematics([0, 0, 0, 0, 0, 0])
h1_left_wrist_pos = left_arm_chain.forward_kinematics([0, 0, 0, 0, 0, 0])
h1_right_wrist_pos = right_arm_chain.forward_kinematics([0, 0, 0, 0, 0, 0])

# inverse kinematics
# front-back, left-right, up-down
for i in range(local_motion_data.shape[0]):
    breakpoint()
    local_pos_left_ankle = local_motion_data[i, human_joint_names.index("left_ankle"), :]
    local_pos_right_ankle = local_motion_data[i, human_joint_names.index("right_ankle"), :]
    local_pos_left_wrist = local_motion_data[i, human_joint_names.index("left_wrist"), :]
    local_pos_right_wrist = local_motion_data[i, human_joint_names.index("right_wrist"), :]

    left_leg_dof = left_leg_chain.inverse_kinematics(local_pos_left_ankle)
    right_leg_dof = right_leg_chain.inverse_kinematics(local_pos_right_ankle)
    left_arm_dof = left_arm_chain.inverse_kinematics(local_pos_left_wrist)
    right_arm_dof = right_arm_chain.inverse_kinematics(local_pos_right_wrist)

    # apply dof and global pos, ori
    h1_dof_positions[1:6] = left_leg_dof[1:]
    h1_dof_positions[6:11] = right_leg_dof[1:]
    h1_dof_positions[11:16] = left_arm_dof[1:]
    h1_dof_positions[16:] = right_arm_dof[1:]
    
    # gym.simulate(sim)
    # gym.fetch_results(sim, True)
    # gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
    
    # gym.step_graphics(sim)
    # gym.sync_frame_time(sim)

print("Done")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


