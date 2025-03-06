file_path = '/cephfs_yili/shared/xuehan/H1_RL/dn_jtroot_18_mdm.pkl'
import joblib
import torch
import numpy as np
# 复制前后帧并 concat
def extend_frames(data_dict, num_frames=20):
    extended_data = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            extended_value = extend_frames(value, num_frames)
        elif isinstance(value, torch.Tensor):
            B = value.shape[0]  # 获取 B，即帧数
            # 复制前面20帧（取第一个帧复制 num_frames 次）
            front = value[0:1].repeat((num_frames,) + (1,) * (value.ndim - 1))
            # 复制后面20帧（取最后一个帧复制 num_frames 次）
            back = value[-1:].repeat((num_frames,) + (1,) * (value.ndim - 1))
            # 拼接 [20 + B + 20] 形状的新 tensor
            extended_value = torch.cat([front, value, back], dim=0)
        else:
            B = value.shape[0]  # 获取 B，即帧数
            # 复制前面20帧（取第一个帧复制 num_frames 次）
            front = np.tile(value[0:1], (num_frames,) + (1,) * (value.ndim - 1))
            # 复制后面20帧（取最后一个帧复制 num_frames 次）
            back = np.tile(value[-1:], (num_frames,) + (1,) * (value.ndim - 1))
            # 拼接 [20 + B + 20] 形状的新数组
            extended_value = np.concatenate([front, value, back], axis=0)
        extended_data[key] = extended_value
    
    return extended_data
data = joblib.load(file_path)
# 使用该函数扩展字典中的数组
extended_data = extend_frames(data)
breakpoint()
extend_path = file_path.split('.pkl')[0] + '_extend.pkl'
joblib.dump(extended_data, extend_path)
