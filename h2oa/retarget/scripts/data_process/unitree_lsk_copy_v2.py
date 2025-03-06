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
from isaacgym.diff_quat import *
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

import matplotlib.pyplot as plt

import math
import numpy as np
import torch
import random
import time
import os
import ipdb
# import ikpy.chain
# import ikpy.utils.plot as plot_utils
import joblib
import sys
from scipy.spatial.transform import Rotation as sRot
sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
# from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, quaternion_to_axis_angle,axis_angle_to_matrix
from tqdm import tqdm
from smpl_sim.utils.torch_ext import to_torch
import matplotlib.pyplot as plt

from phc_h1.utils import torch_utils
show = True

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

def load_motion_with_skeleton(curr_file, skeleton_tree):

    seq_len = curr_file['root_trans_offset'].shape[0]
    start, end = 0, seq_len

    trans = curr_file['root_trans_offset'].clone()[start:end]
    # pose_aa = to_torch(curr_file['pose_aa'][start:end])
    pose_quat_global = curr_file['pose_quat_global'][start:end]
    # pose_quat = curr_file['pose_quat'][start:end]
    

    pose_quat_global = to_torch(pose_quat_global)
    # pose_quat = to_torch(pose_quat)

    sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat_global, trans, is_local=False)

    curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
    return curr_motion



def local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape # torch.Size([259, 24, 4])
    # print(local_rot.shape) 
    
    # dof_pos = quaternion_to_axis_angle(local_rot[:, 1:]) # torch.Size([259, 23, 3])
    # dof_pos = sRot.from_quat(local_rot[:, 1:]).as_euler('yzx')
    dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
    # dof_pos[:,13] *=0
    # dof_pos = torch.zeros_like(torch_utils.quat_to_exp_map(local_rot[:, 1:]))
    # dof_pos[:,14] = torch_utils.quat_to_exp_map(local_rot[:, 15])
    # print(dof_pos)
    # print(dof_pos.shape)
    # print(dof_pos[0])
    return dof_pos.reshape(B, -1)

def local_rotation_to_dof_h1(local_rot,conpen,zero=False):
    B, J, _ = local_rot.shape
    
    # selected_dof[i] : h1[i] - > human[selected_dof[i]]

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
    L_Thorax_rot = selected_rot[:, used_human_joint_names.index('L_Thorax')]
    R_Thorax_rot = selected_rot[:, used_human_joint_names.index('R_Thorax')]
    
    L_Shoulder_rot = selected_rot[:, used_human_joint_names.index('L_Shoulder')]
    L_Elbow_rot = selected_rot[:, used_human_joint_names.index('L_Elbow')]
    R_Shoulder_rot = selected_rot[:, used_human_joint_names.index('R_Shoulder')]
    R_Elbow_rot = selected_rot[:, used_human_joint_names.index('R_Elbow')]

    Torso_rot=sRot.from_quat(Torso_rot)*sRot.from_quat(Spine_rot)*sRot.from_quat(Chest_rot)
    
    Torso_v = Torso_rot.apply([1,0,0])

    Torso_v=torch.from_numpy(Torso_v)
    Torso_euler=torch.atan2(Torso_v[...,1], Torso_v[...,0])[...,None]
    root_pitch=torch.atan2(Torso_v[...,0], Torso_v[...,2])-math.pi/2
    if conpen:
        L_Hip_rot = (R.from_euler('y', -root_pitch)*R(L_Hip_rot)).as_quat()
        R_Hip_rot = (R.from_euler('y', -root_pitch)*R(R_Hip_rot)).as_quat()

    L_Hip_euler = torch.from_numpy(sRot.from_quat(L_Hip_rot).as_euler('yxz'))[:,[2,1,0]]
    L_Knee_euler = torch.from_numpy(sRot.from_quat(L_Knee_rot).as_euler('yzx'))
    L_Ankle_euler = torch.from_numpy(sRot.from_quat(L_Ankle_rot).as_euler('yzx'))
    R_Hip_euler = torch.from_numpy(sRot.from_quat(R_Hip_rot).as_euler('yxz'))[:,[2,1,0]]

    R_Knee_euler = torch.from_numpy(sRot.from_quat(R_Knee_rot).as_euler('yzx'))
    R_Ankle_euler = torch.from_numpy(sRot.from_quat(R_Ankle_rot).as_euler('yzx'))
    
    
    # L_Shoulder_euler = torch.from_numpy(sRot.from_quat(L_Shoulder_rot).as_euler('zxy'))


    theta = math.pi*5/36


    # L_Shoulder_rpy = sRot.from_rotvec([math.pi*5/36,0,0])
    
    # L_Shoulder_v_raw=sRot.from_euler('xy', L_Shoulder_euler[:,1:3]).apply([0,1,0])


    # L_Shoulder_v=sRot.inv(L_Shoulder_rpy).apply(L_Shoulder_v_raw)
    # L_Shoulder_v=torch.from_numpy(L_Shoulder_v)
    # L_Shoulder_v_raw=torch.from_numpy(L_Shoulder_v_raw)
    # L_Shoulder_pitch=-torch_utils.normalize_angle(torch.atan2(L_Shoulder_v[...,2], L_Shoulder_v[...,0])+math.pi/2)
    # L_Shoulder_roll=-torch.arccos(L_Shoulder_v[...,1])+math.pi/2+math.pi*5/36

    # L_Shoulder_x=sRot.from_quat(L_Shoulder_rot).apply([1,0,0])
    # L_Shoulder_x_now=(sRot.from_rotvec(L_Shoulder_pitch[...,None] * torch.tensor([0,math.cos(math.pi*5/36),math.sin(math.pi*5/36)]))*
    #                   sRot.from_euler('x', L_Shoulder_roll)).apply([1,0,0])
    # L_Shoulder_x=torch.from_numpy(L_Shoulder_x)
    # L_Shoulder_x_now=torch.from_numpy(L_Shoulder_x_now)

    qL = (R(L_Thorax_rot)*R(L_Shoulder_rot)*R.from_rotvec([math.pi/2-theta,0,0]))
    L_Shoulder_euler = torch.from_numpy(qL.as_euler('ZXY'))
    L_Shoulder_euler[:,2] = -L_Shoulder_euler[:,2]
    L_Shoulder_euler[:,1] += theta
    
    L_Elbow_angle = +torch.from_numpy(sRot.from_quat(L_Elbow_rot).as_rotvec()[:,2,None]) + math.pi/2

    L_Elbow_v = sRot.from_quat(L_Elbow_rot).apply([0,1,0])
    L_Elbow_v=torch.from_numpy(L_Elbow_v)
    

    L_Shoulder_conpen = torch.from_numpy(sRot.from_quat(L_Elbow_rot).as_rotvec()[:,0]) 

    if conpen:
        L_Shoulder_euler[:,2] += L_Shoulder_conpen #L_Elbow_yaw #  
    # print(L_Shoulder_conpen/math.pi*180)
    # L_Shoulder_euler = torch.stack([L_Shoulder_pitch, L_Shoulder_roll, L_Shoulder_euler[:,0]],dim=-1)

    qR = (R(R_Thorax_rot)*R(R_Shoulder_rot)*R.from_rotvec([theta-math.pi/2,0,0]))
    R_Shoulder_euler = torch.from_numpy(qR.as_euler('ZXY'))
    R_Shoulder_euler[:,0] = -R_Shoulder_euler[:,0]
    R_Shoulder_euler[:,1] -= theta
    



    R_Elbow_angle = -torch.from_numpy(sRot.from_quat(R_Elbow_rot).as_rotvec()[:,2,None]) + math.pi/2

    R_Elbow_v = sRot.from_quat(R_Elbow_rot).apply([0,-1,0])
    R_Elbow_v=torch.from_numpy(R_Elbow_v)
    

    R_Shoulder_conpen = torch.from_numpy(sRot.from_quat(R_Elbow_rot).as_rotvec()[:,0]) 

    if conpen:
        R_Shoulder_euler[:,2] += R_Shoulder_conpen #L_Elbow_yaw # 

   



    dof_pos = torch.cat([L_Hip_euler, L_Knee_euler[:,:1], L_Ankle_euler[:,:1], R_Hip_euler, R_Knee_euler[:,:1], R_Ankle_euler[:,:1], Torso_euler, L_Shoulder_euler, L_Elbow_angle, R_Shoulder_euler, R_Elbow_angle], dim=1)
    
    if zero:
        tmp = dof_pos[:,11:14].clone()
        dof_pos = torch.zeros_like(dof_pos)
        # if conpen:
        dof_pos[:,11:14] = tmp
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
sim_params.dt = dt = 1.0 / 30.0

gymutil.parse_sim_config({"gravity":[0,0,-9.81],"up_axis":1},sim_params)

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    # sim_params.gravity = [0., 0., -9.81]  # [m/s^2]
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
if show:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")


# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# configure env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)


envs = []
actor_handles = []



# load h1 asset
# asset_root = "../HST/isaacgym/assets/"
# h1_asset_file = "h1_description/urdf/h1.urdf"
asset_root = "../HST/legged_gym/resources/robots/h1/urdf"
h1_asset_file = "h1.urdf"
asset_options = gymapi.AssetOptions()
# asset_options.armature = 0.01
asset_options.fix_base_link = False #??????????????????????????????
asset_options.disable_gravity = True
# asset_options.flip_visual_attachments = True
# asset_options.default_dof_drive_mode = 3
asset_options.collapse_fixed_joints = True
# asset_options.replace_cylinder_with_capsule = True
asset_options.flip_visual_attachments = False
# asset_options.fix_base_link = False
# asset_options.density = 0.001
# asset_options.angular_damping = 0
# asset_options.linear_damping = 0
# asset_options.max_angular_velocity = 1000
# asset_options.max_linear_velocity = 1000
# asset_options.armature = 0
# asset_options.thickness = 0.01
# asset_options.disable_gravity = True



h1_asset = gym.load_asset(sim, asset_root, h1_asset_file, asset_options)

h1_dof_names = gym.get_asset_dof_names(h1_asset)
h1_dof_props = gym.get_asset_dof_properties(h1_asset)
h1_num_dofs = gym.get_asset_dof_count(h1_asset)
h1_dof_states = np.zeros(h1_num_dofs, dtype=gymapi.DofState.dtype)


h1_dof_types = [gym.get_asset_dof_type(h1_asset, i) for i in range(h1_num_dofs)]
h1_dof_positions = h1_dof_states['pos']

h1_dof_props["lower"] -=0.2
h1_dof_props["upper"] +=0.2
h1_lower_limits = h1_dof_props["lower"] 
h1_upper_limits = h1_dof_props["upper"] 
h1_ranges = h1_upper_limits - h1_lower_limits
h1_mids = 0.3 * (h1_upper_limits + h1_lower_limits)
h1_stiffnesses = h1_dof_props['stiffness']
h1_dampings = h1_dof_props['damping']
h1_armatures = h1_dof_props['armature']
h1_has_limits = h1_dof_props['hasLimits']

h1_dof_states_v2 = np.zeros(h1_num_dofs, dtype=gymapi.DofState.dtype)
# h1_root_states = np.zeros(13, dtype=gymapi.RigidBodyState.dtype)



# load human motion data
skeleton_tree = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/smpl_humanoid_1.xml")
skeleton_tree_h1 = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/h1_with_wrist.xml")

human_asset = gym.load_asset(sim, '.', "phc_h1/data/assets/mjcf/smpl_humanoid_1.xml", asset_options)

in_file='data/amass/pkls/amass_isaac_train_0.pkl' 
amass_data_list = joblib.load(in_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])

human_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
used_human_joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Torso', 'Spine', 'Chest', 'L_Thorax','L_Shoulder', 'L_Elbow', 'R_Thorax','R_Shoulder', 'R_Elbow']
human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)



for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 1.05)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    # trans = gymapi.Transform()
    # transform.p = (x,y,z)
    # transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
    # gym.set_camera_transform(camera_handle, env, transform)


    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    for t in range(20):
        gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0,0.8))

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1.0, 0, 1.05)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    actor_handle_ref = gym.create_actor(env, human_asset, pose, "actor_ref", i, 1)
    actor_handles.append(actor_handle_ref)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(2.0, 0, 1.05)
    actor_handle2 = gym.create_actor(env, h1_asset, pose, "actor2", i, 1)
    actor_handles.append(actor_handle2)
    for t in range(20):
        gym.set_rigid_body_color(env,actor_handle2,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.6,0,0))


# position the camera
if show:
    cam_pos = gymapi.Vec3(3, -2, 2)
    cam_target = gymapi.Vec3(0, 0, 1)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)
    trans = gym.get_viewer_camera_transform(viewer, envs[0])



# breakpoint()

# 69/3=23
human_dof_names = ['L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 
'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 'R_Ankle_x', 
'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 
'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 
'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 
'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Hand_x', 'L_Hand_y', 'L_Hand_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 
'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 
'R_Wrist_z', 'R_Hand_x', 'R_Hand_y', 'R_Hand_z']
# 19 z x y
h1_dof_names = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 
'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint',
'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'] # get_actor_dof_names
# 20
h1_rigid_body_names = ['pelvis', 
                        'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 
                        'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 
                        'torso_link', 
                        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
                        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'] # gym.get_actor_rigid_body_names

# 19
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

gym.prepare_sim(sim)

gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)

rigid_body_tensor = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
root_state_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
# [2,13] 3 4 3 4
# position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).

net_contact_forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))

# print(root_state_tensor.shape)
root_state_tensor_human = root_state_tensor[1].clone()

# root_state_tensor_zeros = torch.zeros_like(root_state_tensor)
initial_root_rot = [0,0,0.707,0.707]
# initial_root_rot = [-0.5,0.5,0.5,0.5]

h1_motion = []
while True:
    if  show and gym.query_viewer_has_closed(viewer):
        break
    tmp=0
    
    fig, ax = plt.subplots()

    

    for motoin_id, (amass_name, amass_data) in enumerate(amass_data_list.items()):
        # print(amass_name)
        if  show and gym.query_viewer_has_closed(viewer):
            break
        motion = load_motion_with_skeleton(amass_data, skeleton_tree)
        # if tmp==0:
        #     motion = load_motion_with_skeleton(amass_data, skeleton_tree)
        #     tmp=1
        #     continue
        # if tmp==1:
        #     motion = load_motion_with_skeleton(amass_data, skeleton_tree)
        #     tmp=2
        rotlist = []
        line, = ax.plot(rotlist)
        global_translation = motion.global_translation #local_motion_data
        # print(global_translation.shape) # torch.Size([259, 24, 3])
        global_rotation = motion.global_rotation #local_motion_data

        root_transformation = motion.global_transformation[..., 0, :].clone()
        # print(root_transformation.shape) # torch.Size([259, 7]) [rot,trans]

        # root_translation = global_translation[:,0] - global_translation[0,0]
        root_translation = root_transformation[:,4:]
        root_translation[:,0] = root_translation[:,0] - root_translation[0,0] # TODO
        root_translation[:,1] = root_translation[:,1] - root_translation[0,1] # TODO
        root_translation[:,2] = root_translation[:,2] + 0.14

        root_rotatoin = root_transformation[:,:4]
        
        # root_translation = root_translation[:,[0,2,1]] # x, z, y
        # root_translation[:,2] *= -1
        
        dof_smpl = local_rotation_to_dof_smpl(motion.local_rotation) # [259, 69]

        conpen1 = True
        conpen2 = False
        dof_h1, root_pitch = local_rotation_to_dof_h1(motion.local_rotation, conpen=conpen1,zero=False) # [259, 19]

        dof_h1_v2, root_pitch_v2 = local_rotation_to_dof_h1(motion.local_rotation,conpen=conpen2,zero=False) 

        
        
        
        dof_h1_vel= np.zeros_like(dof_h1)
        dof_h1_vel[1: , :] = (dof_h1[1: , :] - dof_h1[:-1, :]) * motion.fps
        dof_h1_vel[0, :]  = dof_h1_vel[1, :]
        
        local_body_info_list = []
        global_body_info_list = []
        # for i in tqdm(range(global_translation.shape[0])):
        for i in (range(global_translation.shape[0])):
            # i=130.
            rotlist.append(dof_h1[i][h1_dof_names.index('left_shoulder_yaw_joint')]/math.pi*180.0)

            ax.cla()
    
            # 重新绘制数据
            line, = ax.plot(rotlist,'-o')
            
            # 刷新图表显示
            plt.pause(0.001)
            # print(i)
            # time.sleep(0.08)s0j
            if  show and gym.query_viewer_has_closed(viewer):
                break
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            # gym.refresh_actor_root_state_tensor(sim)
            # gym.refresh_dof_state_tensor(sim)
            # gym.refresh_rigid_body_state_tensor(sim)
            
            human_dof_states['pos'] = dof_smpl[i] # 69
            h1_dof_states['pos'] = dof_h1[i] # 19
            # h1_dof_states['vel'] = dof_h1_vel[i] # TODO 应该不用设置速度
            h1_dof_states_v2['pos'] = dof_h1_v2[i]
            gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
            gym.set_actor_dof_states(envs[0], actor_handles[1], human_dof_states, gymapi.STATE_POS)
            gym.set_actor_dof_states(envs[0], actor_handles[2], h1_dof_states_v2, gymapi.STATE_POS)

            # gym.set_actor_root_state_tensor_indexed(sim, h1_root_states,actor_handles[0],1) sRot.from_euler('z', -root_pitch[i]) * 
            # print(type(root_translation))
            root_state_tensor[:3, :3] = torch.tensor(R(initial_root_rot).apply(root_translation[i].numpy()))
            root_state_tensor[2, 1] -= 1.5


            # root_state_tensor[:1, 3:7] = torch.from_numpy((R(root_rotatoin[i])*R(initial_root_rot)*sRot.from_euler('y', root_pitch[i])).as_quat())
            
            
            off1 = sRot.from_euler('y', 0)
            if conpen1:
                off1 = sRot.from_euler('y', root_pitch[i])
            off2 = sRot.from_euler('y', 0)
            if conpen2:
                off2 = sRot.from_euler('y', root_pitch[i])
            
            

            root_state_tensor[:1, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])*off1).as_quat())
            root_state_tensor[2, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])*off2).as_quat())

            root_state_tensor[1:2, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])).as_quat())
            
            # root_state_tensor[1] = root_state_tensor_human
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state_tensor))


            # breakpoint()
            # gym.refresh_actor_root_state_tensor(sim)
            # gym.refresh_dof_state_tensor(sim)
            # gym.refresh_rigid_body_state_tensor(sim)

            if show:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.clear_lines(viewer)
                gym.sync_frame_time(sim)

            # breakpoint()
            rigid_body_tensor_copy = rigid_body_tensor.clone().contiguous()
            base_pos = rigid_body_tensor_copy[:1, :3]
            base_ori = rigid_body_tensor_copy[:1, 3:7]
            base_ori_inv = quat_inv(base_ori)

            local_body_pos = quat_apply(base_ori_inv, rigid_body_tensor_copy[:20, :3] - base_pos).view(-1) # 60
            local_body_ori = flip_quat_by_w(quat_multiply(base_ori_inv, rigid_body_tensor_copy[:20, 3:7]))
            local_body_ori = quat_to_vec6d(local_body_ori, do_normalize=True).view(-1) # 120
            local_body_vel = quat_apply(base_ori_inv, rigid_body_tensor_copy[:20, 7:10]).view(-1) # 60
            local_body_ang_vel = quat_apply(base_ori_inv, rigid_body_tensor_copy[:20, 10:]).view(-1) # 60

            local_body_info_list.append(np.concatenate([local_body_pos.numpy(), local_body_ori.numpy(), local_body_vel.numpy(), local_body_ang_vel.numpy()]))


            global_body_pos = rigid_body_tensor_copy[:20, :3].reshape(-1) # 60
            global_body_ori = flip_quat_by_w(rigid_body_tensor_copy[:20, 3:7])
            global_body_ori = quat_to_vec6d(global_body_ori, do_normalize=True).reshape(-1) # 120
            global_body_vel = rigid_body_tensor_copy[:20, 7:10].reshape(-1) # 60
            global_body_ang_vel = rigid_body_tensor_copy[:20, 10:].reshape(-1) # 60
            global_body_info_list.append(np.concatenate([global_body_pos.numpy(), global_body_ori.numpy(), global_body_vel.numpy(), global_body_ang_vel.numpy()]))



        local_motion = np.stack(local_body_info_list)
        global_motion = np.stack(global_body_info_list)
        dof_motion = np.concatenate([dof_h1, dof_h1_vel], axis = -1)

        # dof_pos = np.stack(h1_dof_positions)
        # h1_motion.append(dof_pos)
        amass_name = f'amass_test_{motoin_id}'
        # if motoin_id == 4:
        #     np.save("../HST/isaacgym/h1_motion_data/" + "jt_" + amass_name + ".npy", dof_motion) #[n, 19x2]
        #     np.save("../HST/isaacgym/h1_motion_data/" + "local_" + amass_name + ".npy", local_motion) #[n, (1+19)x(3+6+3+3)]
        #     np.save("../HST/isaacgym/h1_motion_data/" + "global_" + amass_name + ".npy", global_motion) #[n, (1+19)x(3+6+3+3)]
            # break
        
# h1_motion = np.stack(h1_motion)


print("Done")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


