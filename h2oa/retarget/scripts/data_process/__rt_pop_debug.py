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

import matplotlib.pyplot as plt

import math
import numpy as np
import torch
import os
import joblib
import sys
# from scipy.spatial.transform import Rotation as sRot
sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from tqdm import tqdm
# from smpl_sim.utils.torch_ext import to_torch
import matplotlib.pyplot as plt
from time import sleep
from phc_h1.utils import torch_utils

show_plot = False
conpen1 = True
conpen2 = False 
fixed = False
show = False
save = True
show_human = False
def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

def load_motion_with_skeleton(curr_file, skeleton_tree):

    seq_len = curr_file['root_trans_offset'].shape[0]
    start, end = 0, seq_len

    trans = curr_file['root_trans_offset'].clone()[start:end].cpu()
    # pose_aa = to_torch(curr_file['pose_aa'][start:end])
    if 'pose_quat_global' in curr_file.keys():
        pose_quat_global = curr_file['pose_quat_global'][start:end]
        pose_quat_global = to_torch(pose_quat_global)
        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat_global, trans, is_local=False)
    else:
        pose_quat = curr_file['pose_quat'][start:end]
    
        pose_quat = to_torch(pose_quat).cpu()
        # pose_quat = torch.from_numpy((sRot.from_quat(pose_quat.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).reshape(seq_len, -1, 4)  # should fix pose_quat as well here...
        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, pose_quat, trans, is_local=False)

    # pose_quat = to_torch(pose_quat)

    fps = curr_file.get("fps", 30)
    if fps!=30:
        breakpoint()
    curr_motion = SkeletonMotion.from_skeleton_state(sk_state, fps)
    return curr_motion



def local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape # torch.Size([259, 24, 4])
    dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
    return dof_pos.reshape(B, -1)


import pytorch_kinematics as pk
chain = pk.build_chain_from_urdf(open("../HST/legged_gym/resources/robots/h1/urdf/h1_add_hand_link_for_pk.urdf","rb").read())

human_link_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 
                     'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 
                     'Spine', 'Chest', 'Neck', 'Head', 
                     'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 
                     'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
# used_human_joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Torso', 'Spine', 'Chest', 'L_Thorax','L_Shoulder', 'L_Elbow', 'R_Thorax','R_Shoulder', 'R_Elbow']

h1_rigid_body_names = ['pelvis', 
                       'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 
                       'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 
                       'torso_link',
                       'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_link', 
                       'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_link'] # gym.get_actor_rigid_body_names


joint_correspondence = [
    [h1_rigid_body_names.index("pelvis"), human_link_names.index("Pelvis"), 0,1],
    
    # [h1_rigid_body_names.index("left_hip_yaw_link"), human_link_names.index("L_Hip")],
    # [h1_rigid_body_names.index("left_hip_roll_link"), human_link_names.index("L_Hip")],
    # [h1_rigid_body_names.index("left_hip_pitch_link"), human_link_names.index("L_Hip")],
    [h1_rigid_body_names.index("left_knee_link"), human_link_names.index("L_Knee"), 1],
    [h1_rigid_body_names.index("left_ankle_link"), human_link_names.index("L_Ankle"), 1],

    # [h1_rigid_body_names.index("right_hip_yaw_link"), human_link_names.index("R_Hip")],
    # [h1_rigid_body_names.index("right_hip_roll_link"), human_link_names.index("R_Hip")],
    # [h1_rigid_body_names.index("right_hip_pitch_link"), human_link_names.index("R_Hip")],
    [h1_rigid_body_names.index("right_knee_link"), human_link_names.index("R_Knee"), 1],
    [h1_rigid_body_names.index("right_ankle_link"), human_link_names.index("R_Ankle"), 1],

    [h1_rigid_body_names.index("left_shoulder_pitch_link"), human_link_names.index("L_Thorax"), 1],
    [h1_rigid_body_names.index("left_shoulder_roll_link"), human_link_names.index("L_Shoulder"), 1],
    # [h1_rigid_body_names.index("left_shoulder_yaw_link"), human_link_names.index("L_Shoulder")],
    [h1_rigid_body_names.index("left_elbow_link"), human_link_names.index("L_Elbow"), 1],
    [h1_rigid_body_names.index("left_hand_link"), human_link_names.index("L_Hand"), 1],

    [h1_rigid_body_names.index("right_shoulder_pitch_link"), human_link_names.index("R_Thorax"), 1],
    [h1_rigid_body_names.index("right_shoulder_roll_link"), human_link_names.index("R_Shoulder"), 1],
    [h1_rigid_body_names.index("right_elbow_link"), human_link_names.index("R_Elbow"), 1],
    [h1_rigid_body_names.index("right_hand_link"), human_link_names.index("R_Hand"), 1],
]
def forward_kinematics(joint_angle, global_rotvec=None, global_translation=None, device="cuda:0"):
    """
    chain: pytorch_kinematics.chain.Chain
    link_names: list, len = 25
    joint_angle: (N, 19) 19D vector
    global_translation: (N, 3) 3D vector, root_to_world
    global_orientation: (N, 3) 3D axis-angle, root_to_world

    return: a dict contains the global poses of 25 links
    """

    N_frame = joint_angle.shape[0]
    R = vec6d_to_matrix(global_rotvec)  # (N_frame, 3, 3)
    t = global_translation.reshape(N_frame, 3, 1)  # (N_frame, 3, 1)
    root_to_world = torch.cat((torch.cat((R, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(N_frame, 1, 1).to(device)), dim=1)  # (N_frame, 4, 4)

    link_to_root_dict = chain.forward_kinematics(joint_angle)  # link to root
    link_to_world_dict = {}
    for link_name in h1_rigid_body_names:
        T = link_to_root_dict[link_name].get_matrix()  # link to root
        link_to_world_dict[link_name] = torch.einsum('bij,bjk->bik', root_to_world, T)
    
    return link_to_world_dict



class H1_Motion_Model(nn.Module):
    def __init__(self, batch_size=1, global_rotations=None, global_translations=None, device="cuda:0"):
        super(H1_Motion_Model, self).__init__()

        self.batch_size = batch_size
        self.device = device

        # N * (3 + 3 + 19) DoF's solution space

        if global_rotations is None:
            self.global_rotations = nn.Parameter(torch.eye(batch_size, 3, 2).to(device), requires_grad=True)  # (N, 3)  # NOTE: 需要接近最优解的初始化
        else:
            self.global_rotations = nn.Parameter(global_rotations.to(device), requires_grad=True)  # (N, 3)
        if global_translations is None:
            self.global_translations = nn.Parameter(torch.zeros(batch_size, 3).to(device), requires_grad=True)  # (N, 3)
        else:
            self.global_translations = nn.Parameter(global_translations.to(device), requires_grad=True)  # (N, 3)
        self.joint_angles = nn.Parameter(torch.zeros(batch_size, 19).to(device), requires_grad=True)  # (N, 19)
    
    def forward(self):
        return {
            "global_rotations": self.global_rotations,
            "global_translations": self.global_translations,
            "joint_angles": self.joint_angles,    
        }
def retarget_smplx_to_h1(gt_joint_positions, root_rot, root_trans, device):

    N_frame = gt_joint_positions.shape[0]

    ########################## start optimization #################################
    h1_motion_model = H1_Motion_Model(N_frame, quat_to_vec6d(root_rot), root_trans, device=device)
    # h1_motion_model = H1_Motion_Model(N_frame, quat_to_rotvec(root_rot), root_trans, device=device)

    optimizer = torch.optim.Adam(h1_motion_model.parameters(), lr=5e-2)
    h1_motion_model.train()

    for epoch in range(2000):
        h1_motion = h1_motion_model()

        pred_link_to_world_dict = forward_kinematics(h1_motion["joint_angles"], global_rotvec=h1_motion["global_rotations"], global_translation=h1_motion["global_translations"], device=device)

        joint_global_position_loss = 0
        for joint_corr in joint_correspondence:
            joint_global_position_loss += ((pred_link_to_world_dict[h1_rigid_body_names[joint_corr[0]]][:, :3, 3] - gt_joint_positions[:, joint_corr[1]])**2).sum(dim=-1).mean()
        
        pred_joint_angles = h1_motion["joint_angles"]
        pred_joint_velocities = pred_joint_angles[1:] - pred_joint_angles[:-1]
        pred_joint_accelerations = pred_joint_velocities[1:] - pred_joint_velocities[:-1]
        pred_root_linear_velocities = pred_link_to_world_dict["pelvis"][1:, :3, 3] - pred_link_to_world_dict["pelvis"][:-1, :3, 3]
        pred_root_linear_acceleration = pred_root_linear_velocities[1:] - pred_root_linear_velocities[:-1]
        joint_local_velocity_loss = pred_joint_velocities.abs().sum(dim=-1).mean()
        joint_local_acceleration_loss = pred_joint_accelerations.abs().sum(dim=-1).mean()
        root_global_linear_acceleration_loss = (pred_root_linear_acceleration**2).sum(dim=-1).mean()

        # TODO: add joint rotation loss

        # TODO: add contact loss

        loss = 1.0 * joint_global_position_loss + 1.0 * joint_local_velocity_loss + 0.0 * root_global_linear_acceleration_loss

        if epoch % 100 == 0:
            print(epoch, loss.item(), joint_global_position_loss.item(), joint_local_velocity_loss.item(), root_global_linear_acceleration_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ########################## finish optimization ################################

    h1_motion = h1_motion_model()
    return h1_motion


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
chain = chain.to(dtype=torch.float32, device=device)

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
# h1_asset_file = "h1_description/urdf/h1_add_hand_link"
asset_root = "../HST/legged_gym/resources/robots/h1/urdf"
# h1_asset_file = "h1.urdf"
h1_asset_file = "h1_add_hand_link.urdf"
asset_options = gymapi.AssetOptions()
# asset_options.armature = 0.01
asset_options.fix_base_link = False #??????????????????????????????
asset_options.disable_gravity = True
# asset_options.flip_visual_attachments = True
# asset_options.default_dof_drive_mode = 3
asset_options.collapse_fixed_joints = False
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

# h1_dof_states_v2 = np.zeros(h1_num_dofs, dtype=gymapi.DofState.dtype)
# h1_root_states = np.zeros(13, dtype=gymapi.RigidBodyState.dtype)



# load human motion data
skeleton_tree = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/smpl_humanoid_1.xml")
# skeleton_tree_h1 = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/h1_with_wrist.xml")

human_asset = gym.load_asset(sim, '.', "phc_h1/data/assets/mjcf/smpl_humanoid_1.xml", asset_options)

in_file='/cephfs_yili/shared/xuehan/H1_RL/amass_train_13912.pkl' 
# in_file='/home/ubuntu/data/PHC/data/amass/pkls/amass_isaac_train_0.pkl' 
amass_data_list = joblib.load(in_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)



env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)

# add actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0, 1.05)
actor_handle = gym.create_actor(env, h1_asset, pose, "actor", 0, 1)
actor_handles.append(actor_handle)


# set default DOF positions
gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    
rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle)
h1_rigid_body_num = len(rigid_body_names)

for t in range(h1_rigid_body_num):
    gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0,0.8))

#############################human smpl avatar################################
if show_human:
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1.0, 0, 1.05)
    actor_handle_ref = gym.create_actor(env, human_asset, pose, "actor_ref", 0, 1)
    actor_handles.append(actor_handle_ref)
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

# print(root_state_tensor.shape)
# root_state_tensor_human = root_state_tensor[1].clone()

# root_state_tensor_zeros = torch.zeros_like(root_state_tensor)
initial_root_rot = [0,0,0.707,0.707]
# initial_root_rot = [-0.5,0.5,0.5,0.5]

h1_motion = []

if show_plot:
    fig, ax = plt.subplots()


id = 0
cnt = 0

names_50 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2165_0.5_names.pkl')
names_32 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2020_0.32_names.pkl')
names_25 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1949_0.25_names.pkl')
names_21 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1803_0.21_names.pkl')
names_18 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_1474_0.18_names.pkl')
names_15 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2036_0.15_names.pkl')
motion_pkls = {}
for motoin_id, (amass_name, amass_data) in tqdm(enumerate(amass_data_list.items()),total=len(amass_data_list)):
    # if motoin_id>1:
    #     break
    # print(amass_name)
    if amass_name not in names_25:
        continue
    if amass_name in names_21:
        continue
    print(str(id)+ ':' +str(amass_name))
    id+=1
    if show and gym.query_viewer_has_closed(viewer):
        break
    motion = load_motion_with_skeleton(amass_data, skeleton_tree)


    global_translation = motion.global_translation.clone()
    root_transformation_xh = motion.root_transformation_xh.clone()

    motion_len = global_translation.shape[0]

    global_translation[:] = global_translation[:,:,[1,0,2]] # -y, x, z, 
    global_translation[:,:,0] = -global_translation[:,:,0]

    global_translation[:,:,0] = global_translation[:,:,0] - global_translation[0,0,0]
    global_translation[:,:,1] = global_translation[:,:,1] - global_translation[0,0,1]
    root_translation = global_translation[:,0]
    root_rotatoin = root_transformation_xh[:,:4]

    dof_smpl = local_rotation_to_dof_smpl(motion.local_rotation) # [259, 69]

    h1_motion = retarget_smplx_to_h1(global_translation.to(device), root_rotatoin.to(device), root_translation.to(device), device)
    
    dof_h1 = h1_motion["joint_angles"].to('cpu').detach().numpy()
    root_h1_quat = vec6d_to_quat(h1_motion["global_rotations"]).to('cpu').detach()
    # root_h1_quat = quat_from_rotvec(h1_motion["global_rotations"]).to('cpu').detach()
    root_h1_trans = h1_motion["global_translations"].to('cpu').detach()

    dof_h1_vel= np.zeros_like(dof_h1)
    dof_h1_vel[1: , :] = (dof_h1[1: , :] - dof_h1[:-1, :]) * motion.fps
    dof_h1_vel[0, :]  = dof_h1_vel[1, :]
    root_state_list = []
    local_body_info_list = []
    global_body_info_list = []
    
    for i in (range(global_translation.shape[0])):
        
        if  show and gym.query_viewer_has_closed(viewer):
            break
        # gym.simulate(sim)
        # sleep(0.05)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        
        human_dof_states['pos'] = dof_smpl[i] # 69
        h1_dof_states['pos'] = dof_h1[i] # 19
        h1_dof_states['vel'] = dof_h1_vel[i] # TODO 应该不用设置速度

        root_state_tensor[0, :3] = root_h1_trans[i]
        root_state_tensor[0, 3:7] = root_h1_quat[i]

        gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
        if show_human:
            gym.set_actor_dof_states(envs[0], actor_handles[1], human_dof_states, gymapi.STATE_POS)
            root_state_tensor[1, :3] = root_translation[i]
            root_state_tensor[1, 3:7] = root_rotatoin[i]

        
        # root_state_tensor[1] = root_state_tensor_human
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

        # breakpoint()
        rigid_body_tensor_copy = rigid_body_tensor.clone().contiguous()
        global_body_info_list.append(rigid_body_tensor_copy[:h1_rigid_body_num, :7])
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
    print(global_motion.shape)
    if save:
        motion_pkls[amass_name] = {
            "global": global_motion,
            "jt": dof_motion
        }
        print(f'saving {amass_name}')
        
    
filename = f'pop_{len(motion_pkls)}.pkl' 
# joblib.dump(motion_pkls, f"/home/ubuntu/data/PHC/{filename}")
joblib.dump(motion_pkls, f"/cephfs_yili/shared/xuehan/H1_RL/{filename}")
print(len(motion_pkls.keys()))
# h1_motion = np.stack(h1_motion)


print("Done")
if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


