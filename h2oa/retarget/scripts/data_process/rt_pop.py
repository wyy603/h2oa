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


import matplotlib.pyplot as plt

import math
import numpy as np
import torch
import os
import joblib
import sys
# from scipy.spatial.transform import Rotation as sRot
sys.path.append(os.getcwd())
from poselib.diff_quat import *
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from h2oa.utils import *
from tqdm import tqdm
# from smpl_sim.utils.torch_ext import to_torch
import matplotlib.pyplot as plt
from time import sleep

show_plot = False
conpen1 = True
conpen2 = False 
fixed = False
show = False
save = True
show_human = False
# set random seed
np.random.seed(42)
torch.set_printoptions(precision=4, sci_mode=False)
# acquire gym interface
device = 'cuda:0'
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



import pytorch_kinematics as pk
chain = pk.build_chain_from_urdf(open(LEGGED_GYM_RESOURCES / "robots/h1/urdf/h1_add_hand_link_for_pk.urdf","rb").read())
chain = chain.to(dtype=torch.float32, device=device)

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
    link_to_world_dict = []
    for link_name in h1_rigid_body_names:
        T = link_to_root_dict[link_name].get_matrix()  # link to root
        link_to_world_dict.append(torch.einsum('bij,bjk->bik', root_to_world, T))
    link_to_world_dict = torch.stack(link_to_world_dict, dim=1) # (22, N_frame, 4, 4)
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
def retarget_smplx_to_h1(gt_joint_positions, root_rot, root_trans):

    N_frame = gt_joint_positions.shape[0]

    ########################## start optimization #################################
    h1_motion_model = H1_Motion_Model(N_frame, quat_to_vec6d(root_rot), root_trans, device=device)
    # h1_motion_model = H1_Motion_Model(N_frame, quat_to_rotvec(root_rot), root_trans, device=device)

    optimizer = torch.optim.Adam(h1_motion_model.parameters(), lr=5e-2)
    h1_motion_model.train()

    for epoch in range(2000):
        h1_motion = h1_motion_model()

        pred_link_global = forward_kinematics(h1_motion["joint_angles"], global_rotvec=h1_motion["global_rotations"], global_translation=h1_motion["global_translations"], device=device)

        joint_global_position_loss = 0
        for joint_corr in joint_correspondence:
            joint_global_position_loss += ((pred_link_global[:, joint_corr[0]][:, :3, 3] - gt_joint_positions[:, joint_corr[1]])**2).sum(dim=-1).mean()
        
        pred_joint_angles = h1_motion["joint_angles"]
        pred_joint_velocities = pred_joint_angles[1:] - pred_joint_angles[:-1]
        # pred_joint_accelerations = pred_joint_velocities[1:] - pred_joint_velocities[:-1]
        # pred_root_linear_velocities = pred_link_to_world_dict["pelvis"][1:, :3, 3] - pred_link_to_world_dict["pelvis"][:-1, :3, 3]
        # pred_root_linear_acceleration = pred_root_linear_velocities[1:] - pred_root_linear_velocities[:-1]
        joint_local_velocity_loss = pred_joint_velocities.abs().sum(dim=-1).mean()
        # joint_local_acceleration_loss = pred_joint_accelerations.abs().sum(dim=-1).mean()
        # root_global_linear_acceleration_loss = (pred_root_linear_acceleration**2).sum(dim=-1).mean()

        # TODO: add joint rotation loss

        # TODO: add contact loss

        loss = 1.0 * joint_global_position_loss + 1.0 * joint_local_velocity_loss #+ 0.0 * root_global_linear_acceleration_loss

        if epoch % 100 == 0:
            print(epoch, loss.item(), joint_global_position_loss.item(), joint_local_velocity_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ########################## finish optimization ################################

    h1_motion = h1_motion_model()
    return h1_motion





# load human motion data
skeleton_tree = SkeletonTree.from_mjcf(SMPL_SIM_DATA / "assets/mjcf/smpl_humanoid_1.xml")
# skeleton_tree_h1 = SkeletonTree.from_mjcf("phc_h1/data/assets/mjcf/h1_with_wrist.xml")

h2o_file = DATASET / "H1_RT/rcp_140_test.pkl"
# in_file='/cephfs_yili/shared/xuehan/PHC/data/amass/pkls/amass_isaac_bio.pkl' 
in_file = DATASET / "H1_RT/amass_15.pkl"
# in_file='/cephfs_yili/shared/xuehan/PHC/data/amass/pkls/amass_isaac_bio.pkl' 
# in_file='/home/ubuntu/data/PHC/data/amass/pkls/amass_isaac_train_0.pkl' 
h2o_names = list(joblib.load(h2o_file).keys()) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
#print(h2o_names)
amass_data_list0 = joblib.load(in_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])
amass_data_list = {k:v for k,v in amass_data_list0.items() if k in h2o_names}

h1_rigid_body_num = 22

# names_50 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/50_2165_names.pkl')
# names_32 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/32_2020_names.pkl')
# names_25 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/25_1949_names.pkl')
# names_21 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/21_1803_names.pkl')
# names_18 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/18_1474_names.pkl')
# names_15 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/15_2036_names.pkl')

motion_pkls = {}
for motoin_id, (amass_name, amass_data) in tqdm(enumerate(amass_data_list.items()),total=len(amass_data_list)):
    # if motoin_id>1:
    #     break
    # print(amass_name)
    # if 'walk' not in amass_name:
    #     continue
    # if amass_name in names_50 or amass_name in names_32 or amass_name in names_25 or amass_name in names_21 or amass_name in names_18 or amass_name in names_15:
    #     continue
    # if amass_name in names_32:
    #     continue
    print(amass_name)
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

    h1_motion = retarget_smplx_to_h1(global_translation.to(device), root_rotatoin.to(device), root_translation.to(device))
    
    dof_h1 = h1_motion["joint_angles"].to('cpu').detach().numpy()
    # root_h1_quat = vec6d_to_quat(h1_motion["global_rotations"]).to('cpu').detach()
    # # root_h1_quat = quat_from_rotvec(h1_motion["global_rotations"]).to('cpu').detach()
    # root_h1_trans = h1_motion["global_translations"].to('cpu').detach()

    dof_h1_vel= np.zeros_like(dof_h1)
    dof_h1_vel[1: , :] = (dof_h1[1: , :] - dof_h1[:-1, :]) * motion.fps
    dof_h1_vel[0, :]  = dof_h1_vel[1, :]
    
    pred_link_global = forward_kinematics(h1_motion["joint_angles"], global_rotvec=h1_motion["global_rotations"], global_translation=h1_motion["global_translations"], device=device)
    
    global_pos = pred_link_global[:, :, :3, 3].to('cpu').detach()
    global_ori = vec6d_to_quat(pred_link_global[:, :, :3, :2]).to('cpu').detach()
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
        
    
filename = f'pop_{len(motion_pkls)}_h2o.pkl' 
# joblib.dump(motion_pkls, f"/home/ubuntu/data/PHC/{filename}")
joblib.dump(motion_pkls, DATASET / f"H1_RT/{filename}")
print(len(motion_pkls.keys()))
# h1_motion = np.stack(h1_motion)


print("Done")


