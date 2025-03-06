import numpy as np
import torch
# import ipdb
import glob
from h2oa.utils import DATASET

def load_target_jt(device, file):
    one_target_jt = np.load(f"../isaacgym/h1_motion_data/{file}").astype(np.float32)
    one_target_jt = torch.from_numpy(one_target_jt).to(device)
    target_jt = one_target_jt.unsqueeze(0)
    size = torch.tensor([one_target_jt.shape[0]]).to(device)
    target_jt_pos, target_jt_vel = target_jt[:, :, :19], target_jt[:, :, 19:]
    return target_jt_pos, target_jt_vel, size

def load_target_jt_concat(device, file):
    import joblib
    # filter_tag = file.split('pkl')[-1]
    data_dict_ori={}
    for file in glob.glob(file):
        data_dict_ori.update(joblib.load(file))
    ###############
    # names_50 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/50_2165_names.pkl')
    # names_32 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/32_2020_names.pkl')
    # names_25 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/25_1949_names.pkl')
    # names_21 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/21_1803_names.pkl')
    # names_18 = joblib.load('/home/ubuntu/data/PHC/18_1474_names.pkl')
    # names_15 = joblib.load('/home/ubuntu/data/PHC/15_2036_names.pkl')
    if 'MDM' not in file:
        names_h2o = list(joblib.load('/cephfs_yili/shared/xuehan/H1_RL/h2o_8204_rename.pkl').keys()) # NOTE
        data_dict = {}
        for name, data in data_dict_ori.items():
            if name in names_h2o:# and name not in names_32:
                data_dict[name] = data
        del data_dict_ori
    else:
        data_dict = data_dict_ori
    #############
    target_jt = []
    target_global = []
    target_length = []
    
    id =0
    for name, data in data_dict.items():
        # if 'pose' not in name: # NOTE
        #     continue
        # if 'hand' not in name: # NOTE
        #     continue
        # if id == 11:
        #     continue
        # if '0-ACCAD_Female1Walking_c3d_B21_s2_-_put_down_box_to_walk_stageii' not in name and '0-SOMA_soma_subject1_sit_001_stageii' not in name:
        #     continue
        one_target_jt = torch.from_numpy(data['jt'])[..., :19]#.to(device)
        one_target_global = torch.from_numpy(data['global'])#.to(device)
        # if one_target_jt.shape[0] < 100:
        #     continue
        target_jt.append(one_target_jt)
        target_global.append(one_target_global)
        target_length.append(one_target_jt.shape[0])
        id +=1
    target_jt = torch.cat(target_jt, dim=0).to(torch.float32).to(device)
    target_global = torch.cat(target_global, dim=0).to(torch.float32).to(device)
    target_length = torch.tensor(target_length, dtype=torch.long).to(device)
    start_id = torch.zeros_like(target_length, dtype=torch.long)
    start_id[1:] = torch.cumsum(target_length[:-1], dim=0)
    print(file, start_id.shape)
    return target_jt, target_global, target_length, start_id#.to('cpu')

def load_target_body(device, file):
    one_target_body = np.load(f"../isaacgym/h1_motion_data/{file}").astype(np.float32)
    one_target_body = torch.from_numpy(one_target_body).to(device)
    target_body = one_target_body.unsqueeze(0)
    target_body_pos, target_body_ori, target_body_vel, target_body_ang_vel = target_body[:, :, :60], target_body[:, :, 60:180], target_body[:, :, 180:240], target_body[:, :, 240:]
    return target_body_pos, target_body_ori, target_body_vel, target_body_ang_vel

def load_target_root(device, file):
    one_target_root = np.load(f"../isaacgym/h1_motion_data/{file}").astype(np.float32)
    one_target_root = torch.from_numpy(one_target_root).to(device)
    target_root = one_target_root.unsqueeze(0)
    size = one_target_root.shape
    target_root_pos, target_root_ori, target_root_vel, target_root_ang_vel = torch.reshape(target_root[:, :, :, :3], (size[0], size[1], -1)), torch.reshape(target_root[:, :, :, 3:7], (size[0], size[1], -1)), torch.reshape(target_root[:, :, :, 7:10], (size[0], size[1], -1)), torch.reshape(target_root[:, :, :, 10:13], (size[0], size[1], -1))
    return target_root_pos, target_root_ori, target_root_vel, target_root_ang_vel

def load_target_pkl(device, file):
    import joblib
    data_list = joblib.load(file)
    target_jt = []
    target_global = []
    target_length = []
    for name, data in data_list.items():
        one_target_jt = torch.from_numpy(data['jt'])#.to(device)
        one_target_global = torch.from_numpy(data['global'])#.to(device)
        target_jt.append(one_target_jt)
        target_global.append(one_target_global)
        target_length.append(one_target_jt.shape[0])
    target_length = torch.tensor(target_length, dtype=torch.long).to(device)
    return target_jt, target_global, target_length,

def load_target_pkl_concat(device, file):
    #afasdasd
    
    import joblib
    # filter_tag = file.split('pkl')[-1]
    #print("load_target_pkl_concat", device, file)
    data_dict_ori={}
    for file in glob.glob(file):
        data_dict_ori.update(joblib.load(file))
    ###############
    # names_50 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/50_2165_names.pkl')
    # names_32 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/32_2020_names.pkl')
    # names_25 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/25_1949_names.pkl')
    # names_21 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/21_1803_names.pkl')
    # names_18 = joblib.load('/home/ubuntu/data/PHC/18_1474_names.pkl')
    # names_15 = joblib.load('/home/ubuntu/data/PHC/15_2036_names.pkl')

    # if 'MDM' not in file:
    #     names_h2o = list(joblib.load(DATASET / 'H1_RT/rcp_140_test.pkl').keys()) # NOTE
    #     data_dict = {}
    #     for name, data in data_dict_ori.items():
    #         if name in names_h2o:# and name not in names_32:
    #             data_dict[name] = data
    #     del data_dict_ori
    # else:
    #     data_dict = data_dict_ori
    data_dict = data_dict_ori

    # breakpoint()
    #############
    target_jt = []
    target_global = []
    target_length = []
    
    id =0
    # breakpoint()
    for name, data in data_dict.items():
        # if 'pose' not in name: # NOTE
        #     continue
        # if 'hand' not in name: # NOTE
        #     continue
        # if id == 11:
        #     continue
        # if '0-ACCAD_Female1Walking_c3d_B21_s2_-_put_down_box_to_walk_stageii' not in name and '0-SOMA_soma_subject1_sit_001_stageii' not in name:
        #     continue
        one_target_jt = torch.from_numpy(data['jt'])#.to(device)
        one_target_global = torch.from_numpy(data['global'])#.to(device)
        # if one_target_jt.shape[0] < 100:
        #     continue
        target_jt.append(one_target_jt)
        target_global.append(one_target_global)
        target_length.append(one_target_jt.shape[0])
        id +=1
    target_jt = torch.cat(target_jt, dim=0).to(torch.float32).to(device)
    target_global = torch.cat(target_global, dim=0).to(torch.float32).to(device)
    target_length = torch.tensor(target_length, dtype=torch.long).to(device)
    start_id = torch.zeros_like(target_length, dtype=torch.long)
    start_id[1:] = torch.cumsum(target_length[:-1], dim=0)
    #print(file, start_id.shape)
    return target_jt, target_global, target_length, start_id#.to('cpu')