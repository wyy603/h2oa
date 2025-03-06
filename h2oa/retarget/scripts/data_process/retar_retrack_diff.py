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
from tqdm import tqdm
import matplotlib.pyplot as plt





show_plot = False
conpen1 = True
conpen2 = False 
fixed = False
show = True
save = False
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



# retarget_file='/home/ubuntu/data/PHC/train_500.pkl'
recycle_file='/home/ubuntu/data/PHC/diff_259.pkl'

# retarget_file='data/amass/pkls/amass_isaac_train_0.pkl' 
# retarget_data_list = joblib.load(retarget_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
# recycle_file='data/amass/pkls/amass_isaac_train_0.pkl' 
recycle_data_list = joblib.load(recycle_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])



for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 1.05)
    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", i, 0)
    actor_handles.append(actor_handle)
    for t in range(20):
        gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0,0.8))

    # pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(1.0, 0, 1.05)
    # actor_handle1 = gym.create_actor(env, h1_asset, pose, "actor", i, 0)
    # actor_handles.append(actor_handle1)
    # for t in range(20):
    #     gym.set_rigid_body_color(env,actor_handle1,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0.6,0.))

    
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

if show_plot:
    fig, ax = plt.subplots()
# with open("example.txt", "w") as file:
#     # print(amass_data_list.keys())
#     file.write(str(retarget_data_list.keys()))


id = 0
cnt = 0

motion_pkls = {}
error_global = 0
error_global_1 = 0
error_per_motion = []
# for (name, data_A), data_B in tqdm(zip(retarget_data_list.items(),recycle_data_list)):
for data_pair in recycle_data_list:
    if data_pair is None:
        continue
    rotlist = []
    # print(amass_name)
    # if 'walk' not in amass_name:
    #     continue
    # name = data_pair['name']
    # print(str(id)+ ':' +name)
    id+=1
    if  show and gym.query_viewer_has_closed(viewer):
        break
    

    
    target_jt_A = torch.from_numpy(data_pair['jt_root_diff'])[:,:19]#.to(device)
    target_global_pos_A = torch.from_numpy(data_pair['jt_root_diff'])[:,19:19+3]#.to(device)
    target_global_ori_A = torch.from_numpy(data_pair['jt_root_diff'])[:,22:22+6]#.to(device)
    target_jt_B = torch.from_numpy(data_pair['jt_root_B'])[:,:19]#.to(device)
    target_global_pos_B = torch.from_numpy(data_pair['jt_root_B'])[:,19:19+3]#.to(device)
    target_global_ori_B = torch.from_numpy(data_pair['jt_root_B'])[:,22:22+6]#.to(device)
    # target_global_pos_B = torch.from_numpy(data_pair['global_B'])[:,:3]#.to(device)
    # target_global_ori_B = torch.from_numpy(data_pair['global_B'])[:,20*3:20*3+6]#.to(device)
    # jt_A = data_A['jt']
    # global_A = data_A['global']
    # jt_B = data_B['jt']
    # global_B = data_B['global']
    
    # root_state_list = []
    # local_body_info_list = []
    # global_body_info_list = []
    # for i in tqdm(range(global_translation.shape[0])):
    for i in (range(target_jt_A.shape[0])):
        
        if  show and gym.query_viewer_has_closed(viewer):
            break
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        
        # human_dof_states['pos'] = dof_smpl[i] # 69
        # h1_dof_states['pos'] = dof_h1[i] # 19
        # h1_dof_states['vel'] = dof_h1_vel[i] # TODO 应该不用设置速度
        # h1_dof_states_v2['pos'] = dof_h1_v2[i]
        h1_dof_states["pos"]=target_jt_A[i][:19]
        # h1_dof_states["vel"] = target_jt_A[i][19:]
        h1_dof_states_v2["pos"]=target_jt_B[i][:19]
        # h1_dof_states_v2["vel"]=target_jt_B[i][19:]
        gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
        # gym.set_actor_dof_states(envs[0], actor_handles[1], human_dof_states, gymapi.STATE_POS)
        gym.set_actor_dof_states(envs[0], actor_handles[1], h1_dof_states_v2, gymapi.STATE_POS)

        # gym.set_actor_root_state_tensor_indexed(sim, h1_root_states,actor_handles[0],1) sRot.from_euler('z', -root_pitch[i]) * 
        # print(type(root_translation))
        # smpl_toe_high = global_translation[i,:,-1].min()#(global_translation[i,4,-1] + global_translation[i,8,-1])/2
        # root_state_tensor[:3, :3] = torch.tensor(R(initial_root_rot).apply(root_translation[i].numpy()))
        # root_state_tensor[2, 1] -= 2.5
        # root_state_tensor[0, 1] -= 1.5

        
        # root_state_tensor[:1, 3:7] = torch.from_numpy((R(root_rotatoin[i])*R(initial_root_rot)*sRot.from_euler('y', root_pitch[i])).as_quat())
        
        
        # off1 = sRot.from_euler('y', 0)
        # if conpen1:
        #     off1 = sRot.from_euler('y', root_pitch[i])
        # off2 = sRot.from_euler('y', 0)
        # if conpen2:
        #     off2 = sRot.from_euler('y', root_pitch[i])
        

        root_state_tensor[0, :3] = target_global_pos_B[i]
        root_state_tensor[1, :3] = target_global_pos_B[i]
        # root_state_tensor[1, 2] = root_state_tensor[1, 0] + 2
        root_state_tensor[0, 3:7] = vec6d_to_quat(target_global_ori_B[i].reshape(3,2))
        root_state_tensor[1, 3:7] = vec6d_to_quat(target_global_ori_B[i].reshape(3,2))

        
        # root_state_tensor[1] = root_state_tensor_human
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state_tensor))


        # breakpoint()
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        error_global = abs((data_pair['jt_root_B'][i,19:19+3] - data_pair['jt_root_diff'][i,19:19+3])).mean().item()
        # error_global = abs((data_pair['global_B'][i,:3*20] - rigid_body_tensor[:20, :3].reshape(-1).numpy())).mean().item()
        rotlist.append(error_global)
        
        if show_plot:
            ax.cla()
            # 重新绘制数据
            line, = ax.plot(rotlist,'-o')
        
            # 刷新图表显示
            plt.pause(0.001)

        if show:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.clear_lines(viewer)
            gym.sync_frame_time(sim)
    error_global_1 += abs((data_pair['jt_root_B'][:,19:19+3] - data_pair['jt_root_diff'][:,19:19+3])).mean().item()
    
    # if show_plot:
    error_per_motion.append(sum(rotlist[:])/len(rotlist[:]))
    print(error_per_motion)
    print("error",sum(error_per_motion)/len(error_per_motion))
    print("error_1",error_global_1/id)

    
# joblib.dump(motion_pkls, "/cephfs_yili/backup/xuehan/dataset/H1_RL/train_5000.pkl")
print(len(motion_pkls.keys()))
# h1_motion = np.stack(h1_motion)


print("Done")
if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


