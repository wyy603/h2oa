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
save = True
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
JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
    'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
    'spine1': 3, 'spine2': 6, 'spine3': 9, 'Neck': 12, 'Head': 15,
    'LCollar': 13, 'LShoulder': 16, 'LElbow': 18, 'LWrist': 20, 'LHand': 22,
    'RCollar': 14, 'RShoulder': 17, 'RElbow': 19, 'RWrist': 21, 'RHand': 23,  
}

def load_motion_with_skeleton_exp(curr_file, skeleton_tree):

    
    trans = curr_file['root'].clone().cpu()
    # trans = curr_file[:,-1].clone().cpu()
    trans[:] = trans[:,[2,0,1]] # -x, z, y
    # trans[:,1] = -trans[:,1]
    
    pose_exp = curr_file['opt_pose'].clone().cpu().reshape(-1,24,3)[:,list(JOINT_MAP.values())]
    pose_exp[:] = pose_exp[...,[2,0,1]] # -x, z, y
    # pose_exp[...,0] = -pose_exp[...,0]
    
    # pose_exp = curr_file[:,:-1].clone().cpu()
    pose_quat = torch_utils.exp_map_to_quat(pose_exp)

    pose_quat = to_torch(pose_quat).cpu()
    # pose_quat = torch.from_numpy((sRot.from_quat(pose_quat.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).reshape(seq_len, -1, 4)  # should fix pose_quat as well here...
    sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat, trans, is_local=True)

    # pose_quat = to_torch(pose_quat)

    # fps = curr_file.get("fps", 30)
    # if fps!=30:
    #     breakpoint()
    # fps = 20
    curr_motion = SkeletonMotion.from_skeleton_state(sk_state, 30)
    return curr_motion


def local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape # torch.Size([259, 24, 4])
    dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:] if J == 24 else local_rot)
    return dof_pos.reshape(B, -1)



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

# load human asset and data
skeleton_tree = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/smpl_humanoid_1.xml")

human_asset = gym.load_asset(sim, '.', "phc_h1/data/assets/mjcf/smpl_humanoid_1.xml", asset_options)
human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)

# amass_file='/home/ubuntu/data/AMASS/data/amass/pkls/amass_isaac_train_0.pkl' 
amass_file='/home/ubuntu/data/AMASS/amass_25.pkl' 
# amass_file='/home/ubuntu/data/MDM/mdm_10swingwalk.pkl' 
# amass_file='/home/ubuntu/data/MDM/dn_10_swingwalk_mdm.pkl' 
# amass_file='/home/ubuntu/data/MDM/mdm_10box.pkl' 
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
for i in range(3):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 1.05)
    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", 0, 1)
    actor_handles.append(actor_handle)
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    h1_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle)
    h1_rigid_body_num = len(h1_rigid_body_names)
    h1_metric_names = ['left_ankle_link', 'right_ankle_link', 'left_hand_link', 'right_hand_link']
    h1_metric_id = [h1_rigid_body_names.index(x) for x in h1_metric_names]

    color=(255/255,97/255,3/255) if i==0 else (60/255, 106/255, 106/255)

    for t in range(h1_rigid_body_num):
        gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))

for _ in range(3):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1.0, 0, 1.05)
    actor_handle_ref = gym.create_actor(env, human_asset, pose, "actor_ref", 0, 1)
    actor_handles.append(actor_handle_ref)
    gym.set_actor_dof_states(env, actor_handle_ref, human_dof_states, gymapi.STATE_ALL)
human_node_names = skeleton_tree.node_names
human_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle_ref) # same as human_node_names
human_metric_names = ['L_Ankle', 'R_Ankle', 'L_Hand', 'R_Hand']
human_metric_id = [human_rigid_body_names.index(x) for x in human_metric_names]

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

# for test_id, (test_name, test_data0) in tqdm(enumerate(test_data_dict0.items())):
for test_id in [0,1,6,7,9,13,19,25,47]:
    test_name=list(test_data_dict0.keys())[test_id]
# while(1):
    # test_name='0-BMLhandball_S02_Novice_Trial_upper_left_082_poses'
    # test_name='A_person_walks_with_small_but_fast_steps,_with_arms_swinging.0'
    # test_name='0-KIT_425_walking_fast05_stageii'
    # if 'walk' not in test_name.lower():
    #     continue
    if test_name not in amass_data_list:
        continue
    if test_name not in test_data_dict1:
        continue
    if test_name not in test_data_dict2:
        continue
    test_data0= test_data_dict0[test_name]
    test_data1= test_data_dict1[test_name]
    test_data2= test_data_dict2[test_name]
    test_datas=[test_data0,test_data1,test_data2]
    if  show and gym.query_viewer_has_closed(viewer):
        break
    # if test_name not in names_50:# or test_name in names_32:
    #     continue
    # if test_name not in names_25:# or name in names_32:
    #     continue
    # if test_name in names_21:
    #     continue
    print(test_name)
    amass_data = amass_data_list[test_name]
    test_length =  test_data0['jt'].shape[0]
    test_length_sum += test_length
    # motion = load_motion_with_skeleton_exp(amass_data, skeleton_tree) 
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
    root_translation[:,2] = root_translation[:,2] - 0.03

    global_translation[:] = global_translation[:,:,[1,0,2]] # -y, x, z, 
    global_translation[:,:,0] = -global_translation[:,:,0]

    dof_smpl = local_rotation_to_dof_smpl(motion.local_rotation) # [259, 69]

    dof_h1s=[]
    root_pos_h1s=[]
    root_ori_h1s=[]
    for test_data in test_datas:
        dof_h1 = torch.from_numpy(test_data['jt'][:test_length])[...,:19]#.to(device)
        dof_h1s.append(dof_h1)
        # dof_h1_vel= np.zeros_like(dof_h1)
        # dof_h1_vel[1: , :] = (dof_h1[1: , :] - dof_h1[:-1, :]) * motion.fps
        # dof_h1_vel[0, :]  = dof_h1_vel[1, :]

        if 'globalv2' in test_data:
            assert test_data['globalv2'].shape[-1] == h1_rigid_body_num * 7
            root_pos_h1 = torch.from_numpy(test_data['globalv2'][:test_length])[:,:3] 
            root_ori_h1 = torch.from_numpy(test_data['globalv2'][:test_length])[:,h1_rigid_body_num*3:h1_rigid_body_num*3+4] # NOTE(xh) visualize for check??
        elif 'global' in test_data:
            # assert test_data['global'].shape[-1] == h1_rigid_body_num * 15
            h1_rigid_body_num = test_data['global'].shape[-1] // 15
            root_pos_h1 = torch.from_numpy(test_data['global'][:test_length])[:,:3] 
            root_ori_h1 = torch.from_numpy(test_data['global'][:test_length])[:,h1_rigid_body_num*3:h1_rigid_body_num*3+6] # NOTE(xh) visualize for check??
        else:
            root_pos_h1 = torch.from_numpy(test_data['root'][:test_length])[:,:3] #.to(device)
            root_ori_h1 = torch.from_numpy(test_data['root'][:test_length])[:,3:] #.to(device)
            
        root_pos_h1[:,-1] = root_pos_h1[:,-1] + 0. 
        root_pos_h1s.append(root_pos_h1)
        root_ori_h1s.append(root_ori_h1)
    for i in (range(global_translation.shape[0])):
        sleep(0.05)
        if  show and gym.query_viewer_has_closed(viewer):
            break
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        
        human_dof_states['pos'] = dof_smpl[i] # 69
        for j in range(3):
            gym.set_actor_dof_states(envs[0], actor_handles[-1-j], human_dof_states, gymapi.STATE_POS)

        for j, dof_h1 in enumerate(dof_h1s):
            h1_dof_states['pos'] = dof_h1[i][:19] # 19
            gym.set_actor_dof_states(envs[0], actor_handles[j], h1_dof_states, gymapi.STATE_POS)


        for j, root_pos_h1 in enumerate(root_pos_h1s):
            if j == 1:
                root_state_tensor[1, :3] = torch.tensor(R(initial_root_rot).apply(root_translation[i].numpy()))
                root_state_tensor[1, 2] = root_pos_h1[i,2]
            else:
                root_state_tensor[j, :3] = root_pos_h1[i]


            root_state_tensor[-1-j, :3] = torch.tensor(R(initial_root_rot).apply(root_translation[i].numpy()))
            root_state_tensor[j, 1] += j 
            root_state_tensor[-1-j, 1] += j 
        # root_state_tensor[0, 1] -= 1.5
        
        
        for j, root_ori_h1 in enumerate(root_ori_h1s):
            root_rot_h1_ = vec6d_to_quat(root_ori_h1[i].reshape(3,2)) if root_ori_h1.shape[-1] == 6 else root_ori_h1[i]
            if j==1:
                root_state_tensor[j, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rot_h1_)).as_quat())
            else:
                root_state_tensor[j, 3:7] = root_rot_h1_
        # root_state_tensor[0, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])).as_quat())
            root_state_tensor[-1-j, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])).as_quat())
        
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state_tensor))

        cam_pos = gymapi.Vec3(*((root_state_tensor[1, :3] + np.array([4,0,1])).tolist()))
        # print(R(root_state_tensor[1, 3:7]).apply(np.array([2, 1, 1])))
        cam_target = gymapi.Vec3(*(root_state_tensor[1, :3].tolist()))
        gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

        # breakpoint()
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)

        if show:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.clear_lines(viewer)
            gym.sync_frame_time(sim)
            

