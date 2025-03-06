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
# from scipy.spatial.transform import Rotation
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
from tqdm import tqdm
from smpl_sim.utils.torch_ext import to_torch
from phc_h1.utils import torch_utils
debug = False

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

def calculate_rotation_angles(L_Shoulder_x_now, L_Shoulder_x, L_Shoulder_v):
    """
    计算 L_Shoulder_x_now 绕 L_Shoulder_vs 旋转得到 L_Shoulder_x 的旋转角度。
    角度范围在 [-pi/2, pi/2] 之间。
    
    参数:
    - L_Shoulder_x_now: 当前的肩部位置向量，形状为 (n, 3)
    - L_Shoulder_x: 目标的肩部位置向量，形状为 (n, 3)
    - L_Shoulder_vs: 旋转轴向量，形状为 (n, 3)
    
    返回:
    - 旋转角度，形状为 (n,)
    """
    # 单位化向量
    L_Shoulder_x_now = L_Shoulder_x_now / torch.norm(L_Shoulder_x_now, dim=1, keepdim=True)
    L_Shoulder_x = L_Shoulder_x / torch.norm(L_Shoulder_x, dim=1, keepdim=True)
    L_Shoulder_v = L_Shoulder_v / torch.norm(L_Shoulder_v, dim=1, keepdim=True)
    
    # 计算点积
    dot_product = torch.sum(L_Shoulder_x_now * L_Shoulder_x, dim=1)
    
    # 计算旋转角度
    angle = torch.acos(dot_product)
    
    # 计算叉积
    cross_product = torch.cross(L_Shoulder_x_now, L_Shoulder_x, dim=1)
    
    # 判断叉积的方向
    cross_product_dot = torch.sum(cross_product * L_Shoulder_v, dim=1)
    angle[cross_product_dot < 0] *= -1
    
    # 确保角度在 [-pi/2, pi/2] 之间
    # angle = torch.where(angle > torch.pi / 2, angle - torch.pi, angle)
    # angle = torch.where(angle < -torch.pi / 2, angle + torch.pi, angle)
    
    return angle

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

def load_motion_with_skeleton(motion_data, skeleton_tree):
    # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
    # np.random.seed(np.random.randint(5000)* pid)
    curr_file = motion_data

    seq_len = curr_file['root_trans_offset'].shape[0]
    start, end = 0, seq_len

    trans = curr_file['root_trans_offset'].clone()[start:end]*0
    pose_aa = to_torch(curr_file['pose_aa'][start:end])
    pose_quat_global = curr_file['pose_quat_global'][start:end]
    pose_quat = curr_file['pose_quat'][start:end]
    

    # B, J, N = pose_quat_global.shape


    # trans_fix = 0

    pose_quat_global = to_torch(pose_quat_global)
    pose_quat = to_torch(pose_quat)
    # pose_quat*=0
    # pose_quat[...,-1]+=1
    # pose_quat[:,-4,]*=0
    # pose_quat[:,-4,0]+=0.70710678
    # pose_quat[:,-4,-1]+=0.70710678
    # pose_quat[:,-9,]*=0
    # pose_quat[:,-9,]+=0.5
    # pose_quat[:,-9,1]-=1
    # pose_quat[:,-4,]*=0
    # pose_quat[:,-4,]+=0.5
    # pose_quat[:,-4,1]-=1
    # pose_quat[:,-4]-=0.5
    # pose_quat_global*=0
    # # pose_quat_global-=0.5
    # pose_quat_global[0,-1]=1
    # pose_quat_global[1:,1]=0.707107
    # pose_quat_global[1:,3]=0.707107
    # breakpoint()
    # (Pdb) sRot.from_euler('z', math.pi*0.5).as_quat()
    # array([0.        , 0.        , 0.70710678, 0.70710678])
    # (Pdb) sRot.from_euler('x', math.pi*0.5).as_quat()
    # array([0.70710678, 0.        , 0.        , 0.70710678])
    # (Pdb) sRot.from_euler('y', math.pi*0.5).as_quat()
    # array([0.        , 0.70710678, 0.        , 0.70710678])
    # sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat_global, trans, is_local=False)
    sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat, trans, is_local=True)

    curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
    return curr_motion

def local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape
    dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
    return dof_pos.reshape(B, -1)

def local_rotation_to_dof_h1(local_rot):
    # local_rot=motion.local_rotation
    B, J, _ = local_rot.shape
    # local_rotation_to_root = motion.local_rotation_to_root
    selected_joint = [human_joint_names.index(j) for j in used_human_joint_names]
    selected_rot = local_rot[:, selected_joint]
    L_Hip_rot = selected_rot[:, used_human_joint_names.index('L_Hip')]
    L_Knee_rot = selected_rot[:, used_human_joint_names.index('L_Knee')]
    L_Ankle_rot = selected_rot[:, used_human_joint_names.index('L_Ankle')]
    R_Hip_rot = selected_rot[:, used_human_joint_names.index('R_Hip')]
    R_Knee_rot = selected_rot[:, used_human_joint_names.index('R_Knee')]
    R_Ankle_rot = selected_rot[:, used_human_joint_names.index('R_Ankle')]
    Torso_rot = selected_rot[:, used_human_joint_names.index('Torso')]
    Spine_rot = selected_rot[:, used_human_joint_names.index('Spine')]
    Chest_rot = selected_rot[:, used_human_joint_names.index('Chest')]
    # Torso_rot = local_rotation_to_root[:, human_joint_names.index('Chest')]
    L_Shoulder_rot = selected_rot[:, used_human_joint_names.index('L_Shoulder')]
    L_Elbow_rot = selected_rot[:, used_human_joint_names.index('L_Elbow')]
    R_Shoulder_rot = selected_rot[:, used_human_joint_names.index('R_Shoulder')]
    R_Elbow_rot = selected_rot[:, used_human_joint_names.index('R_Elbow')]

    Torso_rot=sRot.from_quat(Torso_rot)*sRot.from_quat(Spine_rot)*sRot.from_quat(Chest_rot)
    Torso_v = Torso_rot.apply([1,0,0])
    Torso_v=torch.from_numpy(Torso_v)
    Torso_euler=torch.atan2(Torso_v[...,1], Torso_v[...,0])[...,None]
    root_pitch=torch.atan2(Torso_v[...,0], Torso_v[...,2])-math.pi/2

    L_Hip_euler = torch.from_numpy(sRot.from_quat(L_Hip_rot).as_euler('yxz'))[:,[2,1,0]]
    L_Hip_euler[...,-1]-=root_pitch
    L_Knee_euler = torch.from_numpy(sRot.from_quat(L_Knee_rot).as_euler('yzx'))
    L_Ankle_euler = torch.from_numpy(sRot.from_quat(L_Ankle_rot).as_euler('yzx'))
    R_Hip_euler = torch.from_numpy(sRot.from_quat(R_Hip_rot).as_euler('yxz'))[:,[2,1,0]]
    R_Hip_euler[...,-1]-=root_pitch
    R_Knee_euler = torch.from_numpy(sRot.from_quat(R_Knee_rot).as_euler('yzx'))
    R_Ankle_euler = torch.from_numpy(sRot.from_quat(R_Ankle_rot).as_euler('yzx'))


    L_Shoulder_rpy = sRot.from_euler('x', math.pi*5/36)
    L_Shoulder_v_raw=sRot.from_quat(L_Shoulder_rot).apply([0,1,0])
    L_Shoulder_v=sRot.inv(L_Shoulder_rpy).apply(L_Shoulder_v_raw)
    L_Shoulder_v=torch.from_numpy(L_Shoulder_v)
    L_Shoulder_v_raw=torch.from_numpy(L_Shoulder_v_raw)
    L_Shoulder_pitch=-torch_utils.normalize_angle(torch.atan2(L_Shoulder_v[...,2], L_Shoulder_v[...,0])+math.pi/2)
    L_Shoulder_roll=-torch.arccos(L_Shoulder_v[...,1])+math.pi/2+math.pi*5/36

    L_Shoulder_x=sRot.from_quat(L_Shoulder_rot).apply([1,0,0])
    L_Shoulder_x_now=(sRot.from_rotvec(L_Shoulder_pitch[...,None] * torch.tensor([0,math.cos(math.pi*5/36),math.sin(math.pi*5/36)]))*sRot.from_euler('x', L_Shoulder_roll)).apply([1,0,0])
    L_Shoulder_x=torch.from_numpy(L_Shoulder_x)
    L_Shoulder_x_now=torch.from_numpy(L_Shoulder_x_now)
    L_Elbow_v = sRot.from_quat(L_Elbow_rot).apply([0,1,0])
    L_Elbow_v=torch.from_numpy(L_Elbow_v)
    L_Elbow_yaw=torch.atan2(L_Elbow_v[...,2], L_Elbow_v[...,0])
    L_Shoulder_yaw=calculate_rotation_angles(L_Shoulder_x_now,L_Shoulder_x,-L_Shoulder_v_raw)+L_Elbow_yaw
    L_Shoulder_yaw=torch_utils.normalize_angle(L_Shoulder_yaw)

    L_Shoulder_euler=torch.stack([L_Shoulder_pitch,L_Shoulder_roll,L_Shoulder_yaw],dim=-1)
    L_Elbow_euler=-torch.arccos(L_Elbow_v[...,1:2])+math.pi/2



    R_Shoulder_rpy = sRot.from_euler('x', -math.pi*5/36)
    R_Shoulder_v_raw=sRot.from_quat(R_Shoulder_rot).apply([0,-1,0])
    R_Shoulder_v=sRot.inv(R_Shoulder_rpy).apply(R_Shoulder_v_raw)
    R_Shoulder_v=torch.from_numpy(R_Shoulder_v)
    R_Shoulder_v_raw=torch.from_numpy(R_Shoulder_v_raw)
    R_Shoulder_pitch=-torch_utils.normalize_angle(torch.atan2(R_Shoulder_v[...,2], R_Shoulder_v[...,0])+math.pi/2)
    R_Shoulder_roll=torch.arccos(-R_Shoulder_v[...,1])-math.pi/2-math.pi*5/36

    R_Shoulder_x=sRot.from_quat(R_Shoulder_rot).apply([1,0,0])
    R_Shoulder_x_now=(sRot.from_rotvec(R_Shoulder_pitch[...,None] * torch.tensor([0,-math.cos(math.pi*5/36),math.sin(math.pi*5/36)]))*sRot.from_euler('x', R_Shoulder_roll)).apply([1,0,0])
    R_Shoulder_x=torch.from_numpy(R_Shoulder_x)
    R_Shoulder_x_now=torch.from_numpy(R_Shoulder_x_now)
    R_Elbow_v = sRot.from_quat(R_Elbow_rot).apply([0,-1,0])
    R_Elbow_v=torch.from_numpy(R_Elbow_v)
    R_Elbow_yaw=torch.atan2(R_Elbow_v[...,2], R_Elbow_v[...,0])
    R_Shoulder_yaw=calculate_rotation_angles(R_Shoulder_x_now,R_Shoulder_x,-R_Shoulder_v_raw)+R_Elbow_yaw
    R_Shoulder_yaw=torch_utils.normalize_angle(R_Shoulder_yaw)

    R_Shoulder_euler=torch.stack([R_Shoulder_pitch,R_Shoulder_roll,R_Shoulder_yaw],dim=-1)
    R_Elbow_euler=-torch.arccos(-R_Elbow_v[...,1:2])+math.pi/2

    dof_pos = torch.cat([L_Hip_euler, L_Knee_euler[:,:1], L_Ankle_euler[:,:1], R_Hip_euler, R_Knee_euler[:,:1], R_Ankle_euler[:,:1], Torso_euler, L_Shoulder_euler, L_Elbow_euler[:,:1], R_Shoulder_euler, R_Elbow_euler[:,:1]], dim=1)
    return dof_pos.reshape(B, -1), root_pitch.reshape(B, -1)
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
if not debug:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")


# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# configure env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

# position the camera
if not debug:
    cam_pos = gymapi.Vec3(5, 10, 10)
    cam_target = gymapi.Vec3(0,0,0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
envs = []
actor_handles = []



# load h1 asset
asset_root = "/home/ubuntu/workspace/isaacgym/assets/"
h1_asset_file = "h1_description/urdf/h1.urdf"
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

# h1_dof_names = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joright_hand_jointint', 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_hand_joint', 'L_index_proximal_joint', 'L_index_intermediate_joint', 'L_middle_proximal_joint', 'L_middle_intermediate_joint', 'L_pinky_proximal_joint', 'L_pinky_intermediate_joint', 'L_ring_proximal_joint', 'L_ring_intermediate_joint', 'L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'L_thumb_distal_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_hand_joint', 'R_index_proximal_joint', 'R_index_intermediate_joint', 'R_middle_proximal_joint', 'R_middle_intermediate_joint', 'R_pinky_proximal_joint', 'R_pinky_intermediate_joint', 'R_ring_proximal_joint', 'R_ring_intermediate_joint', 'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_intermediate_joint', 'R_thumb_distal_joint']
# joint list for retargeting
true_h1_joint_names = ["torso_joint", "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint", "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"]
true_h1_joint_parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 0, 16, 17, 18, 19]



# load human motion data
skeleton_tree = SkeletonTree.from_mjcf("phc/data/assets/mjcf/smpl_humanoid_1.xml")
skeleton_tree_h1 = SkeletonTree.from_mjcf("phc/data/assets/mjcf/h1_with_wrist.xml")

human_asset = gym.load_asset(sim, '/home/ubuntu/workspace/PHC/', "phc/data/assets/mjcf/smpl_humanoid_1.xml", asset_options)

in_file='data/amass_isaac_test_0.pkl' 
amass_data_list = joblib.load(in_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])

human_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
used_human_joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Torso', 'Spine', 'Chest', 'L_Shoulder', 'L_Elbow', 'R_Shoulder', 'R_Elbow']



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

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1.0, 1.08, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    actor_handle_ref = gym.create_actor(env, human_asset, pose, "actor_ref", i, 1)
    actor_handles.append(actor_handle_ref)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    



human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)
# breakpoint()
h1_motion = []
greenballgeo = gymutil.WireframeSphereGeometry(radius=0.12,color = (0, 1, 0))
redballgeo = gymutil.WireframeSphereGeometry(radius=0.1)
blueballgeo = gymutil.WireframeSphereGeometry(radius=0.15,color=(0, 0, 1))



'''['L_Hip_x'0, 'L_Hip_y'1, 'L_Hip_z'2, 'L_Knee_x'3, 'L_Knee_y'4, 'L_Knee_z'5, 'L_Ankle_x'6, 'L_Ankle_y'7, 'L_Ankle_z'8, 
'L_Toe_x'9, 'L_Toe_y'10, 'L_Toe_z'11, 'R_Hip_x'12, 'R_Hip_y'13, 'R_Hip_z'14, 'R_Knee_x'15, 'R_Knee_y'16, 'R_Knee_z', 'R_Ankle_x', 
'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 
'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 
'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 
'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Hand_x', 'L_Hand_y', 'L_Hand_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 
'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 
'R_Wrist_z', 'R_Hand_x', 'R_Hand_y', 'R_Hand_z'] 69'''

'''['left_hip_yaw_joint'0, 'left_hip_roll_joint'1, 'left_hip_pitch_joint'2, 'left_knee_joint', 'left_ankle_joint', 
'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 
'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'] 19'''

human_dof_names = ['L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 
'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 'R_Ankle_x', 
'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 
'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 
'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 
'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Hand_x', 'L_Hand_y', 'L_Hand_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 
'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 
'R_Wrist_z', 'R_Hand_x', 'R_Hand_y', 'R_Hand_z']

h1_dof_names = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 
'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint',
'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'] # get_actor_dof_names

h1_rigid_body_names = ['pelvis', \
                        'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', \
                        'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', \
                            'torso_link', \
                                'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', \
                                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'] # gym.get_actor_rigid_body_names


dof_map = {
    'left_hip_yaw_joint': 'L_Hip_z',
    'left_hip_roll_joint': 'L_Hip_x',
    'left_hip_pitch_joint': 'L_Hip_y',
    'left_knee_joint': 'L_Knee_y',
    'left_ankle_joint': 'L_Ankle_y',

    'right_hip_yaw_joint': 'R_Hip_z',
    'right_hip_roll_joint': 'R_Hip_x',
    'right_hip_pitch_joint': 'R_Hip_y',
    'right_knee_joint': 'R_Knee_y',
    'right_ankle_joint': 'R_Ankle_y',

    'torso_joint': 'Torso_z',

    'left_shoulder_pitch_joint': 'L_Shoulder_x',
    'left_shoulder_roll_joint': 'L_Shoulder_y',
    'left_shoulder_yaw_joint': 'L_Shoulder_z',
    'left_elbow_joint': 'L_Elbow_y',

    'right_shoulder_pitch_joint': 'R_Shoulder_z',
    'right_shoulder_roll_joint': 'R_Shoulder_x',
    'right_shoulder_yaw_joint': 'R_Shoulder_y',
    'right_elbow_joint': 'R_Elbow_y',
}
selected_dof = [human_dof_names.index(dof_map[name]) for name in h1_dof_names]


for amass_name, amass_data in amass_data_list.items():
    # amass_data = amass_data_list['0-Transitions_mazen_c3d_jumpingjacks_longjump_stageii']
    # pose_aa=amass_data['pose_aa']
    # batch_size = pose_aa.shape[0]
    # smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in human_joint_names if q in SMPL_BONE_ORDER_NAMES]
    # pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
    # pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()
    # pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(-1, 24, 4)
    # root_trans_offset=torch.zeros_like(torch.from_numpy(amass_data['trans']))
    # new_sk_state = SkeletonState.from_rotation_and_root_translation(
    #     skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
    #     torch.from_numpy(pose_quat),
    #     root_trans_offset, # move motion data to origin
    #     is_local=True)


    # pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(-1, 24, 4)  # should fix pose_quat as well here...
    # new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)

    # local_motion_data = new_sk_state.global_translation
    motion = load_motion_with_skeleton(amass_data, skeleton_tree)
    # breakpoint()
    local_motion_data = motion.global_translation #local_motion_data
    dof_smpl = local_rotation_to_dof_smpl(motion.local_rotation) # [259, 24, 4]
    dof_h1, root_pitch = local_rotation_to_dof_h1(motion.local_rotation) # [259, 24, 4]




    for i in tqdm(range(local_motion_data.shape[0])):

        
        gym.simulate(sim)
        # breakpoint()
        # gym.fetch_results(sim, True)
        # human_dof_states['pos'] = pose_aa[i][3:]
        # dof_pos = lrs[i]

        # human_dof_states['pos'] = pose_aa_mj.reshape(batch_size,-1)[i][3:]
        human_dof_states['pos'] = dof_smpl[i]
        h1_dof_states['pos'] = dof_h1[i]
        gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
        gym.set_actor_dof_states(envs[0], actor_handles[1], human_dof_states, gymapi.STATE_POS)
        # gym.set_actor_root_state_tensor_indexed(sim, )

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        # from IPython import embed;embed()


        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        # some visualization
        # joint_pose = gym.get_actor_joint_transforms(env, actor_handle)
        # joint_names = gym.get_actor_joint_names(env, actor_handle)

        # joint_pose_h = gym.get_actor_joint_transforms(env, actor_handle_ref)
        # joint_names_h = gym.get_actor_joint_names(env, actor_handle_ref)

        # link_pose = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))[:25]
        # link_pose_h = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))[25:]
        # link_names = gym.get_actor_rigid_body_names(env, actor_handle)
        # link_names_h = gym.get_actor_rigid_body_names(env, actor_handle_ref)
        
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(link_pose[0][0],link_pose[0][1],link_pose[0][2])

        # gymutil.draw_lines(greenballgeo, gym, viewer, envs[0], pose)

        
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(link_pose[link_names.index('torso_link')][0],
        #                     link_pose[link_names.index('torso_link')][1],
        #                     link_pose[link_names.index('torso_link')][2])

        # gymutil.draw_lines(greenballgeo, gym, viewer, envs[0], pose)

        # torso_joint = joint_pose[joint_names.index('torso_joint')][0]
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(torso_joint[0],torso_joint[1],torso_joint[2])

        # gymutil.draw_lines(redballgeo, gym, viewer, envs[0], pose)

        # root_pos = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(root_pos[0,0],root_pos[0,1],root_pos[0,2])

        # gymutil.draw_lines(blueballgeo, gym, viewer, envs[0], pose)


        dof_pos = np.stack(h1_dof_positions)
        h1_motion.append(dof_pos)
    # breakpoint()
h1_motion = np.stack(h1_motion)


print("Done")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


