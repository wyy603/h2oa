

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
import json

from phc_h1.utils import torch_utils

show_plot = False
conpen1 = True
conpen2 = False 
fixed = False
show = False
save = False
device = 'cuda:0'
def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

def load_motion_with_skeleton_remap(curr_file, skeleton_tree, retarget_json):

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
    source_motion = SkeletonMotion.from_skeleton_state(sk_state, fps)

    joint_mapping = retarget_json["joint_mapping"]
    rotation_to_target_skeleton = torch.tensor(retarget_json["rotation"])

    source_tpose = SkeletonState.from_file(retarget_json["source_tpose"])
    target_tpose = SkeletonState.from_file(retarget_json["target_tpose"])
    # run retargeting
    # breakpoint()
    target_motion = source_motion.retarget_to_by_tpose(
      joint_mapping=retarget_json["joint_mapping"],
      source_tpose=source_tpose,
      target_tpose=target_tpose,
      rotation_to_target_skeleton=rotation_to_target_skeleton,
      scale_to_target_skeleton=retarget_json["scale"]
    )
    frame_beg = retarget_json["trim_frame_beg"]
    frame_end = retarget_json["trim_frame_end"]
    if (frame_beg == -1):
        frame_beg = 0
        
    if (frame_end == -1):
        frame_end = target_motion.local_rotation.shape[0]
        
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
      
    tar_global_pos = target_motion.global_translation[frame_beg:frame_end, ...]
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)
    return source_motion, target_motion


def local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape # torch.Size([259, 24, 4])
    # print(local_rot.shape) 
    
    # dof_pos = quaternion_to_axis_angle(local_rot[:, 1:]) # torch.Size([259, 23, 3])
    # dof_pos = sRot.from_quat(local_rot[:, 1:]).as_euler('yzx')
    dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
    # dof_pos[:,13] *=0
    # dof_pos = torch.zeros_like(torch_utils.quat_to_exp_map(local_rot[:, 1:]))
    # dof_pos[:,20] = torch_utils.quat_to_exp_map(local_rot[:, 21])
    # print(dof_pos
    # print(dof_pos.shape)
    # print(dof_pos[0])
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
# device = args.sim_device if args.use_gpu_pipeline else 'cpu'

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

# h1_dof_states_v2 = np.zeros(h1_num_dofs, dtype=gymapi.DofState.dtype)
# h1_root_states = np.zeros(13, dtype=gymapi.RigidBodyState.dtype)



# load human motion data
skeleton_tree = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/smpl_humanoid_1.xml")
# skeleton_tree_h1 = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/h1_with_wrist.xml")

human_asset = gym.load_asset(sim, '.', "phc_h1/data/assets/mjcf/smpl_humanoid_1.xml", asset_options)

h2o_file = '/cephfs_yili/shared/xuehan/H1_RL/rcp_8204_h2o.pkl'
# in_file='/cephfs_yili/shared/xuehan/H1_RL/amass_train_13912.pkl' 
# in_file='/cephfs_yili/shared/xuehan/H1_RL/amass_17268.pkl' 
# in_file='/cephfs_yili/shared/xuehan/PHC/data/amass/pkls/amass_isaac_bio.pkl' 
in_file='/cephfs_yili/shared/xuehan/PHC/data/amass/pkls/amass_isaac_bio.pkl' 
print('loading')
# h2o_names = list(joblib.load(h2o_file).keys()) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
amass_data_list = joblib.load(in_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
# amass_data_list = {k:v for k,v in amass_data_list.items() if k in h2o_names}
human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)

with open('scripts/assets/retarget_smpl_to_h1.json') as f:
    retarget_json = json.load(f)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 1.05)


    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # for t in range(20):
    #     gym.set_rigid_body_color(env,actor_handle,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0,0.8))

    # pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(1.0, 0, 1.05)
    # # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    # actor_handle_ref = gym.create_actor(env, human_asset, pose, "actor_ref", i, 1)
    # actor_handles.append(actor_handle_ref)

    # # set default DOF positions
    # gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)
    
    # pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(2.0, 0, 1.05)
    # actor_handle2 = gym.create_actor(env, h1_asset, pose, "actor2", i, 1)
    # actor_handles.append(actor_handle2)
    # for t in range(20):
    #     gym.set_rigid_body_color(env,actor_handle2,t,gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.6,0,0))
rigid_body_names=gym.get_actor_rigid_body_names(env, actor_handle)
h1_rigid_body_num = len(rigid_body_names)
# breakpoint()

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
human_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 
                     'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 
                     'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 
                     'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 
                     'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
used_human_joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Torso', 'Spine', 'Chest', 'L_Thorax','L_Shoulder', 'L_Elbow', 'R_Thorax','R_Shoulder', 'R_Elbow']

h1_rigid_body_names = ['pelvis', 
                       'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 
                       'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 
                       'torso_link',
                       'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_link', 
                       'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_link'] # gym.get_actor_rigid_body_names

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
# root_state_tensor_human = root_state_tensor[1].clone()

# root_state_tensor_zeros = torch.zeros_like(root_state_tensor)
initial_root_rot = [0,0,0.707,0.707]
# initial_root_rot = [-0.5,0.5,0.5,0.5]
# initial_root_rot = [0,0,0,1]

h1_motion = []

if show_plot:
    fig, ax = plt.subplots()
# with open("example.txt", "w") as file:
#     # print(amass_data_list.keys())
#     file.write(str(amass_data_list.keys()))


id = 0
cnt = 0

motion_pkls = {}
for motoin_id, (amass_name, amass_data) in tqdm(enumerate(amass_data_list.items()),total=len(amass_data_list)):
    # print(amass_name)
    # if 'walk' not in amass_name:
    #     continue
    print(str(id)+ ':' +str(amass_name))
    id+=1
    if  show and gym.query_viewer_has_closed(viewer):
        break
    source_motion, retarget_motion = load_motion_with_skeleton_remap(amass_data, skeleton_tree, retarget_json)
    # if tmp==0:
    #     motion = load_motion_with_skeleton(amass_data, skeleton_tree)
    #     tmp=1
    #     continue
    # if tmp==1:
    #     motion = load_motion_with_skeleton(amass_data, skeleton_tree)
    #     tmp=2
    if show_plot:
        rotlist = []
        line, = ax.plot(rotlist)
    # global_translation = retarget_motion.global_translation #local_motion_data
    # print(global_translation.shape) # torch.Size([259, 24, 3])
    # global_rotation = retarget_motion.global_rotation #local_motion_data

    root_transformation = source_motion.global_transformation[..., 0, :].clone()
    dof_14 = local_rotation_to_dof_smpl(retarget_motion.local_rotation) # [259, 69]
    # print(root_transformation.shape) # torch.Size([259, 7]) [rot,trans]

    # root_translation = global_translation[:,0] - global_translation[0,0]
    root_translation = root_transformation[:,4:]
    root_translation[:,0] = root_translation[:,0] - root_translation[0,0] # TODO
    root_translation[:,1] = root_translation[:,1] - root_translation[0,1] # TODO
    root_translation[:,2] = root_translation[:,2] + 0.1

    root_rotatoin = root_transformation[:,:4]
    
    # root_translation = root_translation[:,[0,2,1]] # x, z, y
    # root_translation[:,2] *= -1
    
    # dof_smpl = local_rotation_to_dof_smpl(motion.local_rotation) # [259, 69]

    try:
        dof_h1, root_pitch = local_rotation_to_dof_h1(source_motion.local_rotation, conpen=conpen1,zero=False) # [259, 19]
    except:
        continue
    # dof_h1_v2, root_pitch_v2 = local_rotation_to_dof_h1(motion.local_rotation,conpen=conpen2,zero=False) 
    # dof_14[:,[22,31]] = -dof_14[:,[22,31]]
    dof_h1[:,0:11] = dof_14[:,[0,1,2,5,8,9,10,11,14,17,18]] #,23,21,22,25,32,30,31,34
    # dof_h1[:,[11,12,15,16]] = dof_14[:,[23,21,32,30]] #,23,21,22,25,32,30,31,34
    dof_h1[:,[13,17]] = dof_14[:,[22,31]] #,23,21,22,25,32,30,31,34
    
    
    
    dof_h1_vel= np.zeros_like(dof_h1)
    dof_h1_vel[1: , :] = (dof_h1[1: , :] - dof_h1[:-1, :]) * source_motion.fps
    dof_h1_vel[0, :]  = dof_h1_vel[1, :]
    root_state_list = []
    local_body_info_list = []
    global_body_info_list = []
    # for i in tqdm(range(global_translation.shape[0])):
    for i in (range(dof_h1.shape[0])):
        if fixed:
            i=13
        if show_plot:
            rotlist.append(dof_h1[i][h1_dof_names.index('left_shoulder_yaw_joint')]/math.pi*180.0)
            ax.cla()
            # 重新绘制数据
            line, = ax.plot(rotlist,'-o')
        
            # 刷新图表显示
            plt.pause(0.001)
        
        if  show and gym.query_viewer_has_closed(viewer):
            break
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        
        # human_dof_states['pos'] = dof_smpl[i] # 69
        h1_dof_states['pos'] = dof_h1[i] # 19
        h1_dof_states['vel'] = dof_h1_vel[i] # TODO 应该不用设置速度
        # h1_dof_states_v2['pos'] = dof_h1_v2[i]
        gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)
        # gym.set_actor_dof_states(envs[0], actor_handles[1], human_dof_states, gymapi.STATE_POS)
        # gym.set_actor_dof_states(envs[0], actor_handles[2], h1_dof_states_v2, gymapi.STATE_POS)

        # gym.set_actor_root_state_tensor_indexed(sim, h1_root_states,actor_handles[0],1) sRot.from_euler('z', -root_pitch[i]) * 
        # print(type(root_translation))
        # smpl_toe_high = global_translation[i,:,-1].min()#(global_translation[i,4,-1] + global_translation[i,8,-1])/2
        root_state_tensor[:1, :3] = torch.tensor(R(initial_root_rot).apply(root_translation[i].numpy())) # initial_root_rot equal to -y, x, z
        # root_state_tensor[2, 1] -= 2.5
        # root_state_tensor[0, 1] -= 1.5

        
        # root_state_tensor[:1, 3:7] = torch.from_numpy((R(root_rotatoin[i])*R(initial_root_rot)*sRot.from_euler('y', root_pitch[i])).as_quat())
        
        
        off1 = sRot.from_euler('y', 0)
        # if conpen1:
        #     off1 = sRot.from_euler('y', root_pitch[i])
        # off2 = sRot.from_euler('y', 0)
        # if conpen2:
        #     off2 = sRot.from_euler('y', root_pitch[i])
        

        root_state_tensor[:1, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])*off1).as_quat())
        # root_state_tensor[2, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])*off2).as_quat())

        # root_state_tensor[1:2, 3:7] = torch.from_numpy((R(initial_root_rot)*R(root_rotatoin[i])).as_quat())
        
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
        base_pos = rigid_body_tensor_copy[:1, :3]
        base_ori = rigid_body_tensor_copy[:1, 3:7]
        base_ori_inv = quat_inv(base_ori)
        # print(root_state_tensor[:1].shape)
        # print(root_state_tensor.shape)
        # root_state_list.append(root_state_tensor[:1,:7].cpu().clone().numpy())
        # global_body_pos = rigid_body_tensor_copy[:, :3] # 60
        # global_body_ori = rigid_body_tensor_copy[:, 3:7]
        # global_body_ori = quat_to_vec6d(global_body_ori, do_normalize=True).reshape(-1) # 120

        # global_body_vel = torch.zeros_like(global_body_pos)
        # global_body_vel = global_body_pos - global_body_pos
        # global_body_vel = rigid_body_tensor_copy[:20, 7:10].reshape(-1) # 60
        # global_body_ang_vel = rigid_body_tensor_copy[:20, 10:].reshape(-1) # 60
        global_body_info_list.append(rigid_body_tensor_copy[:, :7])


        # local_body_pos = quat_apply(base_ori_inv, rigid_body_tensor_copy[:20, :3] - base_pos).view(-1) # 60
        # local_body_ori = flip_quat_by_w(quat_multiply(base_ori_inv, rigid_body_tensor_copy[:20, 3:7]))
        # local_body_ori = quat_to_vec6d(local_body_ori, do_normalize=True).view(-1) # 120
        # local_body_vel = quat_apply(base_ori_inv, rigid_body_tensor_copy[:20, 7:10]).view(-1) # 60
        # local_body_ang_vel = quat_apply(base_ori_inv, rigid_body_tensor_copy[:20, 10:]).view(-1) # 60
        # local_body_info_list.append(np.concatenate([local_body_pos.numpy(), local_body_ori.numpy(), local_body_vel.numpy(), local_body_ang_vel.numpy()]))
        
        


    # root_state = np.stack(root_state_list)
    # print(root_state.shape)
    # local_motion = np.stack(local_body_info_list)
    global_motion = torch.stack(global_body_info_list)
    global_pos = global_motion[...,:3]
    global_ori = global_motion[...,3:]
    global_vel = np.zeros_like(global_pos)
    global_vel[1:] = (global_pos[1:] - global_pos[:-1])*source_motion.fps
    global_vel[0] = global_vel[1]
    global_ang_vel = np.zeros_like(global_pos)
    diff_quat = quat_multiply(global_ori[1:].reshape(-1,4), quat_inv(global_ori[:-1]).reshape(-1,4))
    
    global_ang_vel[1:] = quat_to_rotvec(diff_quat).reshape(-1,h1_rigid_body_num,3)*source_motion.fps
    global_ang_vel[0] = global_ang_vel[1]
    global_motion = np.concatenate([global_pos.numpy().reshape(-1,h1_rigid_body_num*3), 
        quat_to_vec6d(global_ori).reshape(-1,h1_rigid_body_num,6).numpy().reshape(-1,h1_rigid_body_num*6), 
        global_vel.reshape(-1,h1_rigid_body_num*3), 
        global_ang_vel.reshape(-1,h1_rigid_body_num*3)], axis=-1) # NOTE reshape before concat!!!!!!!
    dof_motion = np.concatenate([dof_h1, dof_h1_vel], axis = -1)
    print(global_motion.shape)
    print(dof_motion.shape)
    breakpoint()
    # dof_pos = np.stack(h1_dof_positions)
    # h1_motion.append(dof_pos)
    # amass_name = f'amass_test_{motoin_id}'
    # if motoin_id == 4:
    # print(dof_h1.shape)
    if save:
        motion_pkls[amass_name] = {
            "global": global_motion,
            "jt": dof_motion
        }
        print(f'saving {amass_name}')
        # np.save("../HST/isaacgym/h1_motion_data/" + "jt_" + amass_name + ".npy", dof_h1) #[n, 19x2]
        # # np.save("../HST/isaacgym/h1_motion_data/" + "dof_vel_" + amass_name + ".npy", dof_h1_vel) #[n, 19x2]
        # # np.save("../HST/isaacgym/h1_motion_data/" + "root_state_" + amass_name + ".npy", root_state) #[n, 19x2]

        # # np.save("../HST/isaacgym/h1_motion_data/walk/" + "body_" + amass_name + ".npy", local_motion) #[n, (1+19)x(3+6+3+3)]
        # np.save("../HST/isaacgym/h1_motion_data/" + "global_" + amass_name + ".npy", global_motion) #[n, (1+19)x(3+6+3+3)]        
    # if cnt==5000:
    #     break
    # cnt+=1
    #     break
    
filename = f'rcphp_{len(motion_pkls)}_{os.path.basename(in_file)}' 
joblib.dump(motion_pkls, f"/cephfs_yili/shared/xuehan/H1_RL/{filename}")
print(len(motion_pkls.keys()))
# h1_motion = np.stack(h1_motion)


print("Done")
if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


