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
from h2oa.utils import *

import matplotlib.pyplot as plt

import math
import numpy as np
import torch
import os
import joblib
import sys
from scipy.spatial.transform import Rotation as sRot
sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from tqdm import tqdm
# from smpl_sim.utils.torch_ext import to_torch
import matplotlib.pyplot as plt

from phc_h1.utils import torch_utils
from time import sleep
show_plot = False
conpen1 = True
conpen2 = False
fixed = False
show = True
# save = True
def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

def load_motion_with_skeleton(curr_file, skeleton_tree, seq_end=-1):
    seq_end = curr_file['root_trans_offset'].shape[0] if seq_end == -1 else seq_end
    start, end = 0, seq_end

    trans = curr_file['root_trans_offset'].clone()[start:end].cpu()
    if 'pose_quat_global' in curr_file.keys():
        pose_quat_global = curr_file['pose_quat_global'][start:end]
        pose_quat_global = to_torch(pose_quat_global)
        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat_global, trans, is_local=False)
    else:
        pose_quat = curr_file['pose_quat'][start:end]
    
        pose_quat = to_torch(pose_quat).cpu()
        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat, trans, is_local=False)

    curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
    return curr_motion



def local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape # torch.Size([259, 24, 4])
    dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:] if J == 24 else local_rot)
    return dof_pos.reshape(B, -1)




def rot_decompose(rot, axis):
    
    axis2 = R.from_quat(rot).apply(axis)

    axis = torch.from_numpy(axis)
    axis2 = torch.from_numpy(axis2)

    theta = torch.arccos(torch.sum(axis[None,:]*axis2,dim=1))

    axis3 = torch.cross(axis2, axis[None,:],dim=1)


    # Normalize axis3 to ensure it's a unit vector
    axis3_norm = torch.norm(axis3, dim=1, keepdim=True)
    # Avoid division by zero by setting zero-norm vectors to a default non-zero vector
    axis3_norm[axis3_norm == 0] = 1e-6
    axis3 = axis3 / axis3_norm
    Rp = axis3 * theta[:,None]

    Ry = R.from_rotvec(Rp) * R.from_quat(rot)
    Rxz = (Ry.inv() * R.from_quat(rot)).as_quat()
    Ry_theta = torch.sum(torch.from_numpy(Ry.as_rotvec())*axis,dim=1)

    return Ry_theta, Rxz

human_joint_names = ['Pelvis', 
                     'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 
                     'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 
                     'Torso', 'Spine', 'Chest', 'Neck', 'Head', 
                     'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 
                     'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

used_human_joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Torso', 'Spine', 'Chest', 'L_Thorax','L_Shoulder', 'L_Elbow', 'R_Thorax','R_Shoulder', 'R_Elbow']


def local_rotation_to_dof_h1(local_rot,conpen,zero=False):
    B, J, _ = local_rot.shape
    
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

    L_Shoulder_rot = (R(L_Thorax_rot)*R(L_Shoulder_rot)).as_quat()
    R_Shoulder_rot = (R(R_Thorax_rot)*R(R_Shoulder_rot)).as_quat()

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

    theta = math.pi*5/36

    L_Shoulder_pitch_rot = R.from_euler('x', math.pi*5/36)
    L_Shoulder_pitch_axis = L_Shoulder_pitch_rot.apply([0,1,0])


    Ryp_theta_L, Rxz = rot_decompose(L_Shoulder_rot, L_Shoulder_pitch_axis)
    Rx_theta_L, Rz = rot_decompose(Rxz, np.array([1,0,0],dtype=np.float64))
    Rz_theta_L, _ = rot_decompose(Rz, np.array([0,0,1],dtype=np.float64))
    
    L_Shoulder_euler = torch.stack([Ryp_theta_L,Rx_theta_L + math.pi/2,Rz_theta_L],dim=-1)


    L_Elbow_angle,L_Elbow_rot2 = rot_decompose(L_Elbow_rot,np.array([0,0,1],dtype=np.float64))

    L_Shoulder_conpen, _ = rot_decompose(L_Elbow_rot2, np.array([1,0,0],dtype=np.float64))
    L_Elbow_angle = L_Elbow_angle[:,None] + math.pi/2

    
    if conpen:
        L_Shoulder_euler[:,2] += L_Shoulder_conpen

    R_Shoulder_pitch_rot = R.from_euler('x', -math.pi*5/36)
    R_Shoulder_pitch_axis = R_Shoulder_pitch_rot.apply([0,1,0])
    

    Ryp_theta_R, Rxz = rot_decompose(R_Shoulder_rot, R_Shoulder_pitch_axis)
    Rx_theta_R, Rz = rot_decompose(Rxz, np.array([1,0,0],dtype=np.float64))
    Rz_theta_R, _ = rot_decompose(Rz, np.array([0,0,1],dtype=np.float64))

    R_Shoulder_euler = torch.stack([Ryp_theta_R,Rx_theta_R - math.pi/2,Rz_theta_R],dim=-1)

    R_Elbow_angle, R_Elbow_rot2 = rot_decompose(R_Elbow_rot,np.array([0,0,-1],dtype=np.float64))

    R_Shoulder_conpen, _ = rot_decompose(R_Elbow_rot2, np.array([1,0,0],dtype=np.float64))
    R_Elbow_angle = R_Elbow_angle[:,None] + math.pi/2


    if conpen:
        R_Shoulder_euler[:,2] += R_Shoulder_conpen

    dof_pos = torch.cat([L_Hip_euler, L_Knee_euler[:,:1], L_Ankle_euler[:,:1], R_Hip_euler, R_Knee_euler[:,:1], R_Ankle_euler[:,:1], Torso_euler, L_Shoulder_euler, L_Elbow_angle, R_Shoulder_euler, R_Elbow_angle], dim=1)
    
    if zero:
        tmp = dof_pos[:,17:19].clone()
        dof_pos = torch.zeros_like(dof_pos)
        # if conpen:
        dof_pos[:,17:19] = tmp
        dof_pos[:,16] = -math.pi/2
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
asset_root = LEGGED_GYM_RESOURCES / "robots/h1/urdf"
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
test_file='/cephfs_yili/shared/xuehan/H1_RL/tk6896_ckpt20000_h2o_8204_rename.pkl'
# test_file='/home/ubuntu/data/PHC/denoise_jtroot_1949_c25test_o8_s819000_retarget_13911_amass_train_13912.pkl'
# test_file='/home/ubuntu/data/PHC/retarget_13911_amass_train_13912.pkl'
# test_file='/cephfs_yili/shared/xuehan/H1_RL/dn_jtroot_11438_25_all__o8_s819000_retarget_13911_amass_train_13912.pkl'
# test_file='/home/ubuntu/data/PHC/retarget_64_amass_train_13912.pkl'
# test_file='/cephfs_yili/shared/xuehan/H1_RL/dn_jtroot_11_mdm.pkl.pkl'
# test_file='/home/ubuntu/data/PHC/recycle_65.pkl'
log_path = os.path.join(os.path.dirname(test_file), 'metrics.csv')
# test_file='/cephfs_yili/shared/xuehan/H1_RL/denoise_jtroot_65.pkl' denoise_jtroot_64_
test_data_dict_ori = joblib.load(test_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])

test_data_dict=test_data_dict_ori
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
print('threshold:', threshold)
print('test_file:', test_file)

# load human asset and data
skeleton_tree = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/smpl_humanoid_1.xml")

human_asset = gym.load_asset(sim, '.', "phc_h1/data/assets/mjcf/smpl_humanoid_1.xml", asset_options)
human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)

# amass_file='/home/ubuntu/data/PHC/data/amass/pkls/amass_isaac_train_0.pkl' 
# amass_file='/home/ubuntu/data/PHC/retarget_fail2464_amass_train_13912.pkl' 
amass_file='/cephfs_yili/shared/xuehan/H1_RL/amass_17268.pkl' 
# amass_file1='/cephfs_yili/shared/xuehan/PHC/data/amass/pkls/amass_isaac_bio.pkl' 
# amass_file2='/cephfs_yili/shared/xuehan/PHC/data/amass/pkls/amass_isaac_train_phc.pkl' 
amass_data_list = joblib.load(amass_file) 
# amass_data_list1 = joblib.load(amass_file1) 
# amass_data_list2 = joblib.load(amass_file2) 
# amass_data_list.update(amass_data_list1)
# amass_data_list.update(amass_data_list2)
# joblib.dump(amass_data_list, f'/cephfs_yili/shared/xuehan/H1_RL/amass_{len(amass_data_list)}.pkl')

envs = []
actor_handles = []

env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)

# add actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0, 1.05)
actor_handle = gym.create_actor(env, h1_asset, pose, "actor", 0, 1)
actor_handles.append(actor_handle)
gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
h1_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle)
h1_rigid_body_num = len(h1_rigid_body_names)
h1_metric_names = ['left_ankle_link', 'right_ankle_link', 'left_hand_link', 'right_hand_link']
h1_metric_id = [h1_rigid_body_names.index(x) for x in h1_metric_names]


for t in range(20):
    gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0,0.8))


pose = gymapi.Transform()
pose.p = gymapi.Vec3(1.0, 0, 1.05)
actor_handle_ref = gym.create_actor(env, human_asset, pose, "actor_ref", 0, 1)
actor_handles.append(actor_handle_ref)
gym.set_actor_dof_states(env, actor_handle_ref, human_dof_states, gymapi.STATE_ALL)
human_node_names = skeleton_tree.node_names
human_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle_ref) # same as human_node_names
human_metric_names = ['L_Ankle', 'R_Ankle', 'L_Hand', 'R_Hand']
human_metric_id = [human_rigid_body_names.index(x) for x in human_metric_names]



# position the camera
if show:
    cam_pos = gymapi.Vec3(3, -2, 2)
    cam_target = gymapi.Vec3(0, 0, 1)
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

h1_motion = []

if show_plot:
    fig, ax = plt.subplots()
# with open("example.txt", "w") as file:
#     # print(amass_data_list.keys())
#     file.write(str(amass_data_list.keys()))


# names_50 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2165_0.5_names.pkl')
# names_32 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2020_0.32_names.pkl')
# names_25 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1949_0.25_names.pkl')
# names_21 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1803_0.21_names.pkl')
# names_18 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1474_0.18_names.pkl')
# names_15 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2036_0.15_names.pkl')
test_length_sum=0

for test_id, (test_name, test_data) in tqdm(enumerate(test_data_dict.items())):
    # if 'walk' not in amass_name:
    #     continue
    if  show and gym.query_viewer_has_closed(viewer):
        break
    # if test_name not in amass_data_list:# or test_name not in names:
    #     continue
    # if test_name not in names_50:# or test_name in names_32:
    #     continue
    # if test_name not in names_25:# or name in names_32:
    #     continue
    # if test_name in names_21:
    #     continue
    amass_data = amass_data_list[test_name]
    test_length =  test_data['jt'].shape[0]
    test_length_sum += test_length
    motion = load_motion_with_skeleton(amass_data, skeleton_tree, test_length) 
    if show_plot:
        rotlist = []
        line, = ax.plot(rotlist)
    global_translation = motion.global_translation

    root_transformation = motion.global_transformation[..., 0, :].clone()
    root_translation = root_transformation[:,4:]
    init_x = root_translation[0,0]
    init_y = root_translation[0,1]

    root_rotatoin = root_transformation[:,:4]
    
    global_translation[:,:,0] = global_translation[:,:,0] - init_x
    global_translation[:,:,1] = global_translation[:,:,1] - init_y
    root_translation[:,0] = root_translation[:,0] - init_x
    root_translation[:,1] = root_translation[:,1] - init_y

    global_translation[:] = global_translation[:,:,[1,0,2]] # -y, x, z, 
    global_translation[:,:,0] = -global_translation[:,:,0]

    dof_smpl = local_rotation_to_dof_smpl(motion.local_rotation) # [259, 69]
    dof_h1_retar, _ = local_rotation_to_dof_h1(motion.local_rotation, conpen=conpen1,zero=False) # [259, 19]

    dof_h1 = torch.from_numpy(test_data['jt'][:test_length])[...,:19]#.to(device)
    dof_h1_vel= np.zeros_like(dof_h1)
    dof_h1_vel[1: , :] = (dof_h1[1: , :] - dof_h1[:-1, :]) * motion.fps
    dof_h1_vel[0, :]  = dof_h1_vel[1, :]

    if 'globalv2' in test_data:
        assert test_data['globalv2'].shape[-1] == h1_rigid_body_num * 7
        root_pos_h1 = torch.from_numpy(test_data['globalv2'][:test_length])[:,:3] 
        root_ori_h1 = torch.from_numpy(test_data['globalv2'][:test_length])[:,h1_rigid_body_num*3:h1_rigid_body_num*3+4] # NOTE(xh) visualize for check??
    elif 'global' in test_data:
        assert test_data['global'].shape[-1] == h1_rigid_body_num * 15
        root_pos_h1 = torch.from_numpy(test_data['global'][:test_length])[:,:3] 
        root_ori_h1 = torch.from_numpy(test_data['global'][:test_length])[:,h1_rigid_body_num*3:h1_rigid_body_num*3+6] # NOTE(xh) visualize for check??
    else:
        root_pos_h1 = torch.from_numpy(test_data['root'][:test_length])[:,:3] #.to(device)
        root_ori_h1 = torch.from_numpy(test_data['root'][:test_length])[:,3:] #.to(device)
    jt_error = dof_h1_retar - dof_h1[...,:19]
    jt_error_mean = jt_error.abs().mean()
    jt_error_all.append(jt_error_mean)
    
    global_body_info_list = []
    delta_1motion=[]
    delta_1motion_max=[]
    for i in (range(global_translation.shape[0])):
        # sleep(0.05)
        if  show and gym.query_viewer_has_closed(viewer):
            break
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        
        human_dof_states['pos'] = dof_smpl[i] # 69
        h1_dof_states['pos'] = dof_h1[i][:19] # 19
        gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
        gym.set_actor_dof_states(envs[0], actor_handles[1], human_dof_states, gymapi.STATE_POS)

        root_state_tensor[0, :3] = root_pos_h1[i]
        root_state_tensor[1, :3] = torch.tensor(R(initial_root_rot).apply(root_translation[i].numpy()))
        # root_state_tensor[0, 1] -= 1.5
        
        

        root_state_tensor[0, 3:7] = vec6d_to_quat(root_ori_h1[i].reshape(3,2)) if root_ori_h1.shape[-1] == 6 else root_ori_h1[i]
        root_state_tensor[1, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])).as_quat())
        
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state_tensor))


        # breakpoint()
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)

        if show:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.clear_lines(viewer)
            gym.sync_frame_time(sim)





