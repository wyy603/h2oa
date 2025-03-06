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
from isaacgym import gymtorch
from isaacgym import gymutil

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# sys.path.append(os.getcwd())
# from smpl_sim.utils.torch_ext import to_torch

show_plot = False
conpen1 = True
conpen2 = False
fixed = False
show = True
save = True
def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
    'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
    'spine1': 3, 'spine2': 6, 'spine3': 9, 'Neck': 12, 'Head': 15,
    'LCollar': 13, 'LShoulder': 16, 'LElbow': 18, 'LWrist': 20, 'LHand': 22,
    'RCollar': 14, 'RShoulder': 17, 'RElbow': 19, 'RWrist': 21, 'RHand': 23,  
}




human_joint_names = ['Pelvis', 
                     'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 
                     'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 
                     'Torso', 'Spine', 'Chest', 'Neck', 'Head', 
                     'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 
                     'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

used_human_joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Torso', 'Spine', 'Chest', 'L_Thorax','L_Shoulder', 'L_Elbow', 'R_Thorax','R_Shoulder', 'R_Elbow']


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
# num_envs = 2
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
# print("Creating %d environments" % num_envs)


asset_options = gymapi.AssetOptions()
# asset_options.armature = 0.01
asset_options.fix_base_link = False 
asset_options.disable_gravity = True
# asset_options.flip_visual_attachments = True
# asset_options.default_dof_drive_mode = 3
asset_options.collapse_fixed_joints = True
# asset_options.replace_cylinder_with_capsule = True
asset_options.flip_visual_attachments = False


# load h1 asset and data
asset_root = "../HST/legged_gym/resources/robots/h1/urdf"
h1_asset_file = "h1_add_hand_link.urdf"

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

threshold = 0.25
# test_file='/home/ubuntu/data/AMASS/dn_1.pkl'
# test_file='/home/ubuntu/data/AMASS/retarget_5000.pkl'
# test_file='/home/ubuntu/data/AMASS/dn_1949_c25_o8_s819000_rt_13911.pkl'
test_file0='/home/ubuntu/data/AMASS/dn_25.pkl'
test_file1='/home/ubuntu/data/AMASS/tkh2o_25.pkl'
test_file2='/home/ubuntu/data/AMASS/tkhp_25.pkl'
# test_file='/home/ubuntu/data/MDM/dn_10_box_mdm.pkl'
# test_file='/home/ubuntu/data/AMASS/tk8089_1105mc8204dn_dn_8198_h2o.pkl'
# test_file='/home/ubuntu/data/PHC/recycle_65.pkl'
# log_path = os.path.join(os.path.dirname(test_file), 'metrics.csv')
# test_file='/cephfs_yili/shared/xuehan/H1_RL/denoise_jtroot_65.pkl' denoise_jtroot_64_
test_data_dict_ori0 = joblib.load(test_file0) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
test_data_dict_ori1 = joblib.load(test_file1) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
test_data_dict_ori2 = joblib.load(test_file2) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])

test_data_dict0=test_data_dict_ori0
test_data_dict1=test_data_dict_ori1
test_data_dict2=test_data_dict_ori2
# names_50 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2165_0.5_names.pkl')
# names_32 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2020_0.32_names.pkl')
# names_25 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1949_0.25_names.pkl')
# names_21 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1803_0.21_names.pkl')
# names_18 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1474_0.18_names.pkl')
# names_15 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2036_0.15_names.pkl')
# test_data_dict = {}
# for name, data in test_data_dict_ori.items():
#     if name in names_50:
#         test_data_dict[name] = data

# save = save if 'jtroot_' in test_file else False
# assert 'jtroot_' in test_file if save else True
# print('saving or not:', save)
# print('threshold:', threshold)
# print('test_file:', test_file)

envs = []
actor_handles = []

env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)

# add actor
for i in range(2):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 1.05)
    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", 0, 1)
    actor_handles.append(actor_handle)
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    h1_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle)
    h1_rigid_body_num = len(h1_rigid_body_names)
    # h1_metric_names = ['left_ankle_link', 'right_ankle_link', 'left_hand_link', 'right_hand_link']
    # h1_metric_id = [h1_rigid_body_names.index(x) for x in h1_metric_names]
    dof_names = gym.get_asset_dof_names(h1_asset)
    num_dofs = len(dof_names)

    # color=(255/255,97/255,3/255) if i==0 else (60/255, 106/255, 106/255)
    color=(1,1,1) if i==0 else (0.5,0.5,0.5)
    # color = (0,0,0)
    for t in range(h1_rigid_body_num):
        gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))

# for t in range(len(human_node_names)):
#     gym.set_rigid_body_color(env,actor_handle_ref,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5,0.2,0.2))


# position the camera
if show:
    # cam_pos = gymapi.Vec3(3, -2, 2)
    # cam_target = gymapi.Vec3(0, 0, 1)

    cam_pos = gymapi.Vec3(1, 4, 1)
    # cam_pos = gymapi.Vec3(5, 7, 2)
    # cam_pos = gymapi.Vec3(-2, 4.5, 2)
    # cam_target = gymapi.Vec3(0, 3.5, 1)
    cam_target = gymapi.Vec3(0, 1, 1)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)
    trans = gym.get_viewer_camera_transform(viewer, envs[0])

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
# root_state_tensor_human = root_state_tensor[1].clone()

# root_state_tensor_zeros = torch.zeros_like(root_state_tensor)
initial_root_rot = [0,0,0.707,0.707]
# initial_root_rot = [-0.5,0.5,0.5,0.5]


dof_h1s=[]
root_pos_h1s=[]
root_ori_h1s=[]
gym.fetch_results(sim, True)
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)

# h1_dof_states['pos'] = dof_h1[i][:19] # 19
stand_joint_angles = {
    'left_hip_yaw_joint': 0.0, # 0
    'left_hip_roll_joint': 0.0,
    'left_hip_pitch_joint': 0,
    'left_knee_joint': 0,
    'left_ankle_joint': -0.,
    'right_hip_yaw_joint': 0.0, # 5
    'right_hip_roll_joint': 0.0,
    'right_hip_pitch_joint': -0.,
    'right_knee_joint': 0,
    'right_ankle_joint': -0.,
    'torso_joint': 0.0, # 10
    'left_shoulder_pitch_joint': 0.,
    'left_shoulder_roll_joint': 3.14,
    'left_shoulder_yaw_joint': 0,
    'left_elbow_joint': 1.57,
    'right_shoulder_pitch_joint': 0., # 15
    'right_shoulder_roll_joint': -3.14,
    'right_shoulder_yaw_joint': 0,
    'right_elbow_joint': 1.57
}
for i in range(num_dofs):
    name = dof_names[i]
    angle = stand_joint_angles[name]
    h1_dof_positions[i] = angle
gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
squat_joint_angles = {
    'left_hip_yaw_joint': 0.0,
    'left_hip_roll_joint': 0.0,
    'left_hip_pitch_joint': -0.9,
    'left_knee_joint': 0,
    'left_ankle_joint': -0,
    'right_hip_yaw_joint': 0.0,
    'right_hip_roll_joint': 0.0,
    'right_hip_pitch_joint': -0.9,
    'right_knee_joint': 0,
    'right_ankle_joint': -0,
    'torso_joint': 0.0,
    'left_shoulder_pitch_joint': -0.94,
    'left_shoulder_roll_joint': 0.21,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': 1.57,
    'right_shoulder_pitch_joint': -0.94,
    'right_shoulder_roll_joint': -0.21,
    'right_shoulder_yaw_joint': 0.0,
    'right_elbow_joint': 1.57
}
for i in range(num_dofs):
    name = dof_names[i]
    angle = squat_joint_angles[name]
    h1_dof_positions[i] = angle
gym.set_actor_dof_states(envs[0], actor_handles[1], h1_dof_states, gymapi.STATE_POS)


# root_state_tensor[0,2]+=111
root_state_tensor[1,0]+=0.15
root_state_tensor[1,2]-=0.05
root_state_tensor[1, 3:7] = torch.from_numpy( R.from_euler('y', 0.9).as_quat())
gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state_tensor))

gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)

# 获取刚体状态
rigid_body_states = gym.acquire_rigid_body_state_tensor(sim)
rigid_body_states = gymtorch.wrap_tensor(rigid_body_states).view(
            2, -1, 13)  # + 1

# 假设 `shoulder_index` 是肩部关节索引
shoulder_names = [s for s in h1_rigid_body_names if ('shoulder' in s and 'roll' in s)]
shoulder_pos=[]
shoulder_pos.append(rigid_body_states[0, h1_rigid_body_names.index(shoulder_names[0]), :3])
shoulder_pos.append(rigid_body_states[0, h1_rigid_body_names.index(shoulder_names[1]), :3])
shoulder_pos.append(rigid_body_states[1, h1_rigid_body_names.index(shoulder_names[0]), :3])
shoulder_pos.append(rigid_body_states[1, h1_rigid_body_names.index(shoulder_names[1]), :3])

# 球面均匀采样（Fibonacci 球面采样）
def fibonacci_sphere(samples=1000, z_range=(0.8, 1), extra_samples=300, mode=0):
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # 黄金角度

    # 正常采样整个球面
    for i in range(samples):
        # z = 1 - (i / float(samples - 1)) * 2  # 从 -1 到 1 的 Z 坐标
        z =  (np.sqrt(i / float(samples - 1))) * 2 - 1
        radius = np.sqrt(1 - z * z) if z>0 else 1 # 计算半径

        theta = phi * i  # 黄金角度

        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        # if z<0.5:
        if mode < 2:
            x -= 0.3*(z-1)
        else:
            x += 0.3*(z-1)

        # 过滤掉右下1/4球面区域（y < 0 且 z < 0）
        if y < -0.2 or z < -0.5:
            continue
        if mode==0:
            points.append((x, y, z))
        elif mode==1:
            points.append((x, -y, z))
        elif mode==2:
            points.append((x, y, -z))
        else:
            points.append((x, -y, -z))



    # 在 z = 0.5 到 0.7 区间内额外采样
    for i in range(extra_samples):
        z = 1. - abs(np.random.randn()*0.2)  # 从 0.5 到 0.7 范围内随机采样 Z 坐标
        radius = np.sqrt(1 - z * z)  # 计算半径

        theta = np.random.uniform(0, 2 * np.pi)  # 在 0 到 2pi 范围内随机选择角度

        x = np.cos(theta) * radius
        y = np.sin(theta) * radius

        # 过滤掉右下1/4球面区域（y < 0 且 z < 0）
        if y < 0 and z < 0:
            continue

        if mode==0:
            points.append((x, y, z))
        elif mode==1:
            points.append((x, -y, z))
        elif mode==2:
            points.append((x, y, -z))
        else:
            points.append((x, -y, -z))

    return np.array(points, dtype=np.float32)

def add_perturbation_to_points(points, perturbation_scale=0.01):
    # 对每个点添加扰动
    perturbed_points = points + np.random.normal(0, perturbation_scale, points.shape)
    return perturbed_points
# 生成球面上的点
num_samples = 500  # 采样点数量
for i in range(4):
    sphere_points = fibonacci_sphere(num_samples,mode=i) * 0.65  # 调整半径
    sphere_points = add_perturbation_to_points(sphere_points)

    sphere_points += shoulder_pos[i].cpu().numpy()

    sphere_geom = gymutil.WireframeSphereGeometry(0.01, 2, 2, None, color=(1, 0, 0) if i%2==0 else (0,0,1))

    # 设置线条（用短线段表示点）
    for i, pos in enumerate(sphere_points):
        # lines[i * 2] = p  # 起点
        # lines[i * 2 + 1] = p + np.array([0.01, 0.01, 0.01], dtype=np.float32)  # 终点
        # colors[i * 2] = colors[i * 2 + 1] = [1, 0, 0]  # 设为红色

        sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)

        gymutil.draw_lines(sphere_geom, gym, viewer, env, sphere_pose)




cam_pos = gymapi.Vec3(*((root_state_tensor[1, :3] + np.array([4,0,1])).tolist()))
# print(R(root_state_tensor[1, 3:7]).apply(np.array([2, 1, 1])))
cam_target = gymapi.Vec3(*(root_state_tensor[1, :3].tolist()))
gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

# breakpoint()
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)
gym.step_graphics(sim)
# gym.clear_lines(viewer)
gym.sync_frame_time(sim)


while(True):
    pass
    if show:
        # gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        # gym.clear_lines(viewer)
        # gym.sync_frame_time(sim)
