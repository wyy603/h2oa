import numpy as np
import torch
import scipy.ndimage.filters as filters
import sys
import os
sys.path.append(os.getcwd())
import poselib.poselib.core.rotation3d as pRot
from diff_quat import vec6d_to_quat, quat_to_vec6d
import joblib
def _compute_velocity(p, time_delta, guassian_filter=True):
    velocity = np.gradient(p.numpy(), axis=-3) / time_delta
    if guassian_filter:
        velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
    else:
        velocity = torch.from_numpy(velocity).to(p)
    
    return velocity

def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
    # assume the second last dimension is the time axis
    diff_quat_data = pRot.quat_identity_like(r).to(r)
    diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :]))
    diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
    angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
    if guassian_filter:
        angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
    return angular_velocity  

# Define the time delta value
fps = 20.
time_delta = 1./fps # Replace with the actual time delta value

in_file='/home/ubuntu/data/PHC/dn_35_mdm.pkl'
ori_motions = joblib.load(in_file)
h1_rigid_body_num = 22
motion_pkls={}
for name, motion in ori_motions.items():
# Compute velocity for global motion
    global_pos = torch.from_numpy(motion['global'][:,:3*h1_rigid_body_num]).reshape(-1,h1_rigid_body_num,3)
    global_ori = torch.from_numpy(motion['global'][:,3*h1_rigid_body_num:9*h1_rigid_body_num]).reshape(-1,h1_rigid_body_num,6)
    global_ori = vec6d_to_quat(global_ori.reshape(-1,3,2)).reshape(-1, h1_rigid_body_num, 4)

    old_vel = torch.from_numpy(motion['global'][:,9*h1_rigid_body_num:12*h1_rigid_body_num]).reshape(-1,h1_rigid_body_num,3)

    global_pos_vel = _compute_velocity(global_pos, time_delta)
    global_ori_ang_vel = _compute_angular_velocity(global_ori, time_delta)

    # Compute velocity for dof motion
    dof_h1 = torch.from_numpy(motion['jt'][:,None,:19])
    old_dof_vel = torch.from_numpy(motion['jt'][:,None,19:])
    dof_h1_vel = _compute_velocity(dof_h1, time_delta)[:,0]

    # Concatenate the re-computed velocities with the original motion arrays
    global_motion = np.concatenate([
        motion['global'][:,:3*h1_rigid_body_num],
        motion['global'][:,3*h1_rigid_body_num:9*h1_rigid_body_num],
        global_pos_vel.reshape(-1, h1_rigid_body_num*3).numpy(),
        global_ori_ang_vel.reshape(-1, h1_rigid_body_num*3).numpy()
    ], axis=-1)

    dof_motion = np.concatenate([
        motion['jt'][:,:19],
        dof_h1_vel.reshape(-1,19).numpy()
    ], axis=-1)
    motion_pkls[name] = {
        "global": global_motion,
        "jt": dof_motion
    }
    print(f'saving {name}')
assert len(motion_pkls) == len(ori_motions)
filename = f'nvel_{os.path.basename(in_file)}' 
joblib.dump(motion_pkls, os.path.join(os.path.dirname(in_file), filename))
print(len(motion_pkls.keys()))