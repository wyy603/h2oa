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

# from isaacgym import gymapi
# from isaacgym import gymutil
# from isaacgym import gymtorch
# from scipy.spatial.transform import Rotation as R

# import matplotlib.pyplot as plt

import math
import numpy as np
import torch
import os
import joblib
import sys
# from scipy.spatial.transform import Rotation as sRot
sys.path.append(os.getcwd())
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from poselib.diff_quat import *
from h2oa.utils import *
from tqdm import tqdm
# from smpl_sim.utils.torch_ext import to_torch
# import matplotlib.pyplot as plt

# from phc_h1.utils import torch_utils

show_plot = False
conpen1 = True
conpen2 = False 
fixed = False
show = False
save = True
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


    assert curr_file.get("fps", 30) == 30
    curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
    return curr_motion



# load human motion data
skeleton_tree = SkeletonTree.from_mjcf(SMPL_SIM_DATA / "assets/mjcf/smpl_humanoid_1.xml")


# in_file='/home/ubuntu/data/PHC/data/amass/pkls/amass_isaac_train_0.pkl' 
in_file='/cephfs_yili/shared/xuehan/PHC/data/amass/pkls/amass_isaac_bio.pkl' 
amass_data_list = joblib.load(in_file) # dict_keys(['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender'])

human_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
used_human_joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Torso', 'Spine', 'Chest', 'L_Thorax','L_Shoulder', 'L_Elbow', 'R_Thorax','R_Shoulder', 'R_Elbow']
# human_dof_states = np.zeros(69, dtype=gymapi.DofState.dtype)







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
h1_rigid_body_names = ['pelvis', 
                        'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 
                        'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 
                        'torso_link', 
                        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
                        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'] # gym.get_actor_rigid_body_names

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





motion_pkls = {}
for motoin_id, (amass_name, amass_data) in tqdm(enumerate(amass_data_list.items())):
    # print(amass_name)
    # if 'walk' not in amass_name:
    #     continue
    print(str(id)+ ':' +str(amass_name))
    # if  show and gym.query_viewer_has_closed(viewer):
    #     break
    motion = load_motion_with_skeleton(amass_data, skeleton_tree)

    local_translation = motion.local_translation.clone() # NOTE:invariant???????????????
    local_rotation = motion.local_rotation.clone()[:,1:] # motion.local_rotation[:,0] should be identical to root_transformation[:,4], different from the root_transformation_xh
    global_translation = motion.global_translation.clone()
    root_transformation_xh = motion.root_transformation_xh.clone()
    # global_rotation = motion.global_rotation.clone()

    motion_len = local_translation.shape[0]

    global_translation[:] = global_translation[:,:,[1,0,2]] # -y, x, z, 
    global_translation[:,:,0] = -global_translation[:,:,0]

    root_translation = root_transformation_xh[:,4:].contiguous().clone()
    root_rotation = root_transformation_xh[:,:4].contiguous().clone()
    init_x = root_translation[0,0]
    init_y = root_translation[0,1]


    global_translation[:,:,0] = global_translation[:,:,0] - init_x
    global_translation[:,:,1] = global_translation[:,:,1] - init_y
    root_translation[:,0] = root_translation[:,0] - init_x
    root_translation[:,1] = root_translation[:,1] - init_y
    # breakpoint()

    root_rotation = quat_to_vec6d(root_rotation).reshape(-1,6)

    local_rotation = quat_to_vec6d(local_rotation).reshape(motion_len,23,6)

    if save:
        motion_pkls[amass_name] = {
            "root_transformation": torch.concat([root_translation,root_rotation], dim=1), # [n, 6+3]
            "local_rotation": local_rotation, # [n, 23, 6]
            "global_translation": global_translation, # [n, 23+1, 3]
        }
        
        print(f'saving {amass_name}')


fp_0='/cephfs_yili/shared/xuehan/H1_RL/human_11113_amass_isaac_train_phc.pkl'
fp_1='/cephfs_yili/shared/xuehan/H1_RL/human_13912_amass_train_13912.pkl'
pk0=joblib.load(fp_0)
pk1=joblib.load(fp_1)
motion_pkls.update(pk0)
motion_pkls.update(pk1)
filename = f'human_{len(motion_pkls.keys())}_amass.pkl' 
joblib.dump(motion_pkls, f"/cephfs_yili/shared/xuehan/H1_RL/{filename}")
print(len(motion_pkls.keys()))
# h1_motion = np.stack(h1_motion)


print("Done")


