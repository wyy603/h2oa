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

# conda activate isaac
# python unitree_h1_retargeting_lsk.py
# import keyboard
import sys
import select

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, axis_angle_to_matrix
from h2oa.utils import *

import math
import numpy as np
import torch
import random
import time
import os
import ipdb
from tqdm import tqdm
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

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
# ipdb.set_trace()
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_fps = 10
sim_params.dt = dt = 1.0 / sim_fps
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

show = True
save = True
save_fps = 10
if show:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

asset_root = str(LEGGED_GYM_RESOURCES / "robots/h1")
#asset_root = str(H2OA_DIR / "diffusion/ddbm/h1/")
#asset_root = str(LEGGED_GYM_RESOURCES / "robots/h1_lower_body")

# load h1 asset
h1_asset_file = "urdf/h1.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
print(sim, asset_root, h1_asset_file, asset_options)
h1_asset = gym.load_asset(sim, asset_root, h1_asset_file, asset_options)
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
h1_dof_props['hasLimits'] = np.array([True]*h1_num_dofs)

num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

envs = []
actor_handles = []
joint_handles = {}

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# load h1 motion data
motoin_name = '0-SOMA_soma_subject2_random_002_stageii.npy'
motion_data_path = os.path.join(DATASET, f"humanplus/h1_motion_data/jt_{motoin_name}")
motion_data = np.load(motion_data_path)[:,:19]
# breakpoint()
offset = np.array([ 0.0000,  0.0000, -0.3490,  0.6980, -0.3490,  0.0000,  0.0000, -0.3490,
          0.6980, -0.3490,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000]) 
motion_cmp = [motion_data + offset, motion_data + offset]

initial_pose_ori = [-0.5,0.5,0.5,0.5]
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.08, 0.0)
    pose.r = gymapi.Quat(*initial_pose_ori)
    
    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)

# position the camera
if show:
    # right view
    # cam_pos = gymapi.Vec3(3, 2.0, 0)
    # cam_target = gymapi.Vec3(-3, 0, 0)
    # front view
    cam_pos = gymapi.Vec3(0, 2.0, -2)
    cam_target = gymapi.Vec3(0, 0, 2)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

gym.prepare_sim(sim)     
for i in tqdm(range(motion_data.shape[0])):
    for i_env in range(num_envs):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        for j in range(motion_data.shape[1]):
            h1_dof_positions[j] = motion_cmp[i_env][i,j]
        gym.set_actor_dof_states(envs[i_env], actor_handles[i_env], h1_dof_states, gymapi.STATE_POS)
    if show:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)
    gym.sync_frame_time(sim)

    if show and gym.query_viewer_has_closed(viewer):
        break

if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

