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
save = False
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
asset_options.collapse_fixed_joints = False
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
# test_file='/home/ubuntu/data/PHC/denoise_new2463_retarget_fail2464_amass_train_13912.pkl'
test_file_paths=[
    '/home/ubuntu/data/PHC/data/amass/pkls/amass_isaac_train_0.pkl',
    '/home/ubuntu/data/PHC/denoise_2161_c25_256_o8_s819000_retarget_13911_amass_train_13912.pkl',
]
test_names=[]
for test_file in test_file_paths:
    test_data_dicts = joblib.load(test_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
    test_names.append(test_data_dicts.keys())
log_path = os.path.join(os.path.dirname(test_file), 'metrics.csv')

if len(test_names) == 1:
    test_names = test_names[0]  # 如果只有一个列表，交集就是它本身
else:
    # 计算所有列表的交集
    test_names = list(set(test_names[0]).intersection(*test_names[1:]))

test_data_dict = {k: [test_data[k] for test_data in test_data_dicts] for k in test_names}


save = save if 'jtroot_' in test_file else False
assert 'jtroot_' in test_file if save else True
print('saving or not:', save)
print('threshold:', threshold)
print('test_file:', test_file)

# load human asset and data
skeleton_tree = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/smpl_humanoid_1.xml")

human_asset = gym.load_asset(sim, '.', "phc_h1/data/assets/mjcf/smpl_humanoid_1.xml", asset_options)
human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)


num_actors = len(test_file_paths)
envs = []
actor_handles = []

env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 1.05)

actor_handle_ref = gym.create_actor(env, human_asset, pose, "actor_ref", 0, 1)
actor_handles.append(actor_handle_ref)
gym.set_actor_dof_states(env, actor_handle_ref, human_dof_states, gymapi.STATE_ALL)

human_node_names = skeleton_tree.node_names
human_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle_ref) # same as human_node_names
human_metric_names = ['L_Ankle', 'R_Ankle', 'L_Hand', 'R_Hand']
human_metric_id = [human_rigid_body_names.index(x) for x in human_metric_names]

for actor_id in range(num_actors-1):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(actor_id+1, 0, 1.05)
    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", 0, 1)
    actor_handles.append(actor_handle)
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    for t in range(22):
        gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0,actor_id/num_actors))

h1_rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle)
h1_rigid_body_num = len(h1_rigid_body_names)
h1_metric_names = ['left_ankle_link', 'right_ankle_link', 'left_hand_link', 'right_hand_link']
h1_metric_id = [h1_rigid_body_names.index(x) for x in h1_metric_names]




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


net_contact_forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))


initial_root_rot = [0,0,0.707,0.707]

h1_motion = []

if show_plot:
    fig, ax = plt.subplots()
    
motion_pkls={}
jt_error_all=[]
test_length_sum=0
endeffector_error_all=[]
endeffector_errormax_all=[]
for test_id, (test_name, test_data_list) in tqdm(enumerate(test_data_dict.items())):
    # if 'walk' not in amass_name:
    #     continue
    print(test_name)
    if  show and gym.query_viewer_has_closed(viewer):
        break
    
    # if test_name not in names_50:# or test_name in names_32:
    #     continue
    # if test_name not in names_25:# or name in names_32:
    #     continue
    # if test_name in names_21:
    #     continue
    if show_plot:
        rotlist = []
        line, = ax.plot(rotlist)

    amass_data = test_data_list[0]
    test_length = amass_data['root_trans_offset'].shape[0]
    test_length_sum += test_length
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

    global_translation[:] = global_translation[:,:,[1,0,2]] # -y, x, z, 
    global_translation[:,:,0] = -global_translation[:,:,0]

    dof_smpl = local_rotation_to_dof_smpl(motion.local_rotation) # [259, 69]

    dof_h1_list = []
    root_pos_h1_list = []
    root_ori_h1_list = []

    for test_data in test_data_list[1:]:
        dof_h1 = torch.from_numpy(test_data['jt'][:test_length])[...,:19]
        dof_h1_list.append(dof_h1)
        if 'global' in test_data:
            assert test_data['global'].shape[-1] == h1_rigid_body_num * 15
            root_pos_h1 = torch.from_numpy(test_data['global'][:test_length])[:,:3] 
            root_ori_h1 = torch.from_numpy(test_data['global'][:test_length])[:,h1_rigid_body_num*3:h1_rigid_body_num*3+6] # NOTE(xh) visualize for check??
        else:
            root_pos_h1 = torch.from_numpy(test_data['root'][:test_length])[:,:3] #.to(device)
            root_ori_h1 = torch.from_numpy(test_data['root'][:test_length])[:,3:] #.to(device)
        root_pos_h1_list.append(root_pos_h1)
        root_ori_h1_list.append(root_ori_h1)
    jt_error = dof_h1_retar - dof_h1[...,:19]
    jt_error_mean = jt_error.abs().mean()
    jt_error_all.append(jt_error_mean)
    
    global_body_info_list = []
    delta_1motion=[]
    delta_1motion_max=[]
    for i in (range(global_translation.shape[0])):
        sleep(0.05)
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
        
        

        root_state_tensor[0, 3:7] = vec6d_to_quat(root_ori_h1[i].reshape(3,2))
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
        if save:
            # root_state_list.append(root_state_tensor[:1,:7].detach().cpu().clone().numpy())
            global_body_info_list.append(rigid_body_tensor[:h1_rigid_body_num, :7].detach().cpu().clone())

        # compute metric
        assert ((global_translation[i,human_metric_id] - rigid_body_tensor[[h1_rigid_body_num+i for i in human_metric_id], :3]).abs() < 0.01).all()
        delta = (global_translation[i,human_metric_id] - rigid_body_tensor[h1_metric_id, :3]).norm(dim=-1).mean()
        delta_max = (global_translation[i,human_metric_id] - rigid_body_tensor[h1_metric_id, :3]).norm(dim=-1).max()
        delta_1motion.append(delta)
        delta_1motion_max.append(delta_max)
        if show_plot:
            rotlist.append(delta.item())
            ax.cla()
            # 重新绘制数据
            line, = ax.plot(rotlist,'-o')
        
            # 刷新图表显示
            plt.pause(0.001)
    delta_1motion_max = torch.tensor(delta_1motion_max).max()
    delta_1motion_mean = torch.tensor(delta_1motion).mean()
    endeffector_error_all.append(delta_1motion_mean)
    endeffector_errormax_all.append(delta_1motion_max)
    if save:
        global_motion = torch.stack(global_body_info_list)
        global_pos = global_motion[...,:3]
        global_ori = global_motion[...,3:]
        global_vel = np.zeros_like(global_pos)
        global_vel[1:] = (global_pos[1:] - global_pos[:-1])*motion.fps
        global_vel[0] = global_vel[1]
        global_ang_vel = np.zeros_like(global_pos)
        diff_quat = quat_multiply(global_ori[1:].reshape(-1,4), quat_inv(global_ori[:-1]).reshape(-1,4))
        
        global_ang_vel[1:] = quat_to_rotvec(diff_quat).reshape(-1,h1_rigid_body_num,3)*motion.fps
        global_ang_vel[0] = global_ang_vel[1]
        global_motion = np.concatenate([global_pos.numpy().reshape(-1,h1_rigid_body_num*3), 
            quat_to_vec6d(global_ori).reshape(-1,h1_rigid_body_num,6).numpy().reshape(-1,h1_rigid_body_num*6), 
            global_vel.reshape(-1,h1_rigid_body_num*3), 
            global_ang_vel.reshape(-1,h1_rigid_body_num*3)], axis=-1) # NOTE reshape before concat!!!!!!!
        dof_motion = np.concatenate([dof_h1, dof_h1_vel], axis = -1)
        # print(global_motion.shape)
        # print(dof_motion.shape)
        
        motion_pkls[test_name] = {
            "global": global_motion,
            "jt": dof_motion
        }
        # print(f'saving {test_name}')
succ_rate = torch.tensor(endeffector_errormax_all) < threshold
endeffector_error_mean=torch.tensor(endeffector_error_all).mean()
jt_error_mean=torch.tensor(jt_error_all).mean()
# breakpoint()
# saving error in log_path(csv)
out_path = test_file.replace("jtroot_", "") if "jtroot_" in test_file else test_file
with open(log_path, "a") as f:
    # write in csv file

    f.write(f"{out_path}, {succ_rate.to(torch.float64).mean():.4g}, {endeffector_error_mean:.4g}, {jt_error_mean:.4g}, /, /, /, /, {test_length_sum/len(test_data_list):.4g}\n")
if save:
    print(out_path)
    joblib.dump(motion_pkls, out_path)
if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


