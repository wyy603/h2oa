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

test_file0='/home/ubuntu/data/AMASS/dn_25.pkl'
test_data_dict_ori0 = joblib.load(test_file0)
test_data_dict0=test_data_dict_ori0

skeleton_tree = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/smpl_humanoid_1.xml")

human_asset = gym.load_asset(sim, '.', "phc_h1/data/assets/mjcf/smpl_humanoid_1.xml", asset_options)
human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)

amass_file='/home/ubuntu/data/AMASS/amass_25.pkl'
amass_data_list = joblib.load(amass_file)

envs = []
actor_handles = []

env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)
actor_num=4

# add actor
for i in range(actor_num):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 1.05)
    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", 0, 1)
    actor_handles.append(actor_handle)
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    h1_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle)
    h1_rigid_body_num = len(h1_rigid_body_names)
    h1_metric_names = ['left_ankle_link', 'right_ankle_link', 'left_hand_link', 'right_hand_link']
    h1_metric_id = [h1_rigid_body_names.index(x) for x in h1_metric_names]

    color=(255/255,97/255,50*i/255)

    for t in range(h1_rigid_body_num):
        gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))

for i in range(actor_num):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1.0, 0, 1.05)
    actor_handle_ref = gym.create_actor(env, human_asset, pose, "actor_ref", 0, 1)
    actor_handles.append(actor_handle_ref)
    gym.set_actor_dof_states(env, actor_handle_ref, human_dof_states, gymapi.STATE_ALL)
    h1_rigid_body_num = len(h1_rigid_body_names)
    # color=(255/255,97/255,3/255) if i==0 else (60/255, 106/255, 106/255)
    #
    # for t in range(3):
    #     gym.set_rigid_body_color(env,actor_handle_ref,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*color))

human_node_names = skeleton_tree.node_names
human_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle_ref) # same as human_node_names

# for t in range(len(human_node_names)):
#     gym.set_rigid_body_color(env,actor_handle_ref,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5,0.2,0.2))


# position the camera
if show:

    cam_pos = gymapi.Vec3(1, 4, 1)
    cam_target = gymapi.Vec3(0, 1, 1)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)
    trans = gym.get_viewer_camera_transform(viewer, envs[0])

gym.prepare_sim(sim)

gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)

rigid_body_tensor = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
root_state_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))


initial_root_rot = [0,0,0.707,0.707]
# initial_root_rot = [-0.5,0.5,0.5,0.5]

h1_motion = []

test_id=47
test_name = list(test_data_dict0.keys())[test_id]
# for test_id, (test_name, test_data0) in enumerate(test_data_dict0.items()):
#     if test_id<46:
#         continue
# for test_id in [0,1,6,7,9,13,19,25,47]:
# for test_id in [3,5,15,31,42]:
#     test_name=list(test_data_dict0.keys())[test_id]
test_data = test_data_dict0[test_name]
amass_data = amass_data_list[test_name]
# amass_data = amass_data_list[test_name]
test_length =  test_data['jt'].shape[0]
print(test_length)
# motion = load_motion_with_skeleton_exp(amass_data, skeleton_tree)
motion = load_motion_with_skeleton(amass_data, skeleton_tree, test_length)

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
root_translation[:,2] = root_translation[:,2] - 0.04 

global_translation[:] = global_translation[:,:,[1,0,2]] # -y, x, z, 
global_translation[:,:,0] = -global_translation[:,:,0]

dof_smpl = local_rotation_to_dof_smpl(motion.local_rotation) # [259, 69]

dof_h1 = torch.from_numpy(test_data['jt'][:test_length])[...,:19]#.to(device)

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
# seq_ids=[50,83,93,103]
seq_ids=range(40,test_length,(test_length-60)//actor_num)
for i in range(actor_num):
    frame_id = seq_ids[i]
    print(frame_id)
    # frame_id = 17 #45
    # gym.fetch_results(sim, True) # NOTE(xh) important!!!!!!!!!!!!!!!!! will change root_state_tensor[i, :3] = root_pos_h1[frame_id]
    # gym.refresh_actor_root_state_tensor(sim)
    # gym.refresh_dof_state_tensor(sim)
    # gym.refresh_rigid_body_state_tensor(sim)
    
    human_dof_states['pos'] = dof_smpl[frame_id] # 69
    gym.set_actor_dof_states(envs[0], actor_handles[actor_num+i], human_dof_states, gymapi.STATE_POS)
    # human_dof_states['pos'][-9] = human_dof_states['pos'][-9] - 0.3
    # human_dof_states['pos'][-8] = human_dof_states['pos'][-8] - 0.3
    # dof_h1[frame_id][4] = dof_h1[frame_id][4] + 0.38
    # dof_h1[frame_id][9] = dof_h1[frame_id][9] - 0.3
    h1_dof_states['pos'] = dof_h1[frame_id][:19] #+torch.rand_like(dof_h1[frame_id][:19])*0.0# 19
    gym.set_actor_dof_states(envs[0], actor_handles[i], h1_dof_states, gymapi.STATE_POS)

    root_state_tensor[i, :3] = root_pos_h1[frame_id]
    root_state_tensor[actor_num+i, :3] = torch.tensor(R(initial_root_rot).apply(root_translation[frame_id].numpy()))
    root_state_tensor[i, 1] += -i*0.3
    root_state_tensor[actor_num+i, 1] +=-i*0.3

    

    root_state_tensor[i, 3:7] = vec6d_to_quat(root_ori_h1[frame_id].reshape(3,2)) if root_ori_h1.shape[-1] == 6 else root_ori_h1[frame_id]
    root_state_tensor[actor_num+i, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[frame_id])).as_quat())

# print(root_state_tensor[:,:3])


gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state_tensor))
    # breakpoint()
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)
while(1):
    if  show and gym.query_viewer_has_closed(viewer):
        break
    if show:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)
        gym.sync_frame_time(sim)
        
sleep(5999999)
# breakpoint()

