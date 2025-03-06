import os.path
import torch
import random
import numpy as np
import torchvision.transforms as transforms
# from .image_folder import make_dataset
from .diff_quat import *
from .diff_rot import *
from PIL import Image

import torchvision
import blobfile as bf

from glob import glob

def get_params( size,  resize_size,  crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}
 

def get_transform(params,  resize_size,  crop_size, method=Image.BICUBIC,  flip=True, crop = True, totensor=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale(img, crop_size, method)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __scale(img, target_width, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        return torch.nn.functional.interpolate(img.unsqueeze(0), size=(target_width, target_width), mode='bicubic', align_corners=False).squeeze(0)
    else:
        return img.resize((target_width, target_width), method)

def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    return __flip(img, flip)


import joblib

class MotionDataset_v0(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, recycle_data_path, retarget_data_path, train=True, human_data_path = None, load_pose = False, norm = False, overlap=-1,window_size=24,mixed_data=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        # data_list_A = joblib.load(data_path_A)
        # data_list_B = joblib.load(data_path_B)
        self.jt_A = []
        self.jt_B = []
        self.jt_C = []
        self.root_A = []
        self.root_B = []
        self.root_C = []
        motion_length = []
        self.names = []
        self.load_pose = load_pose
        h1_num_bodies = 22
        human_num_bodies = 24
        self.window_size = window_size
        self.overlap = overlap
        self.mixed_data = mixed_data
        self.train=train
        # start_id = []
        # current_id = 0
        # for (name, data_A), data_B in zip(data_list_A.items(), data_list_B):
        #     if data_B is not None:
        #         target_jt_A = torch.from_numpy(data_A['jt'])#.to(device)
        #         target_global_pos_A = torch.from_numpy(data_A['global'])[:,:3]#.to(device)
        #         target_global_ori_A = torch.from_numpy(data_A['global'])[:,20*3:20*3+6]#.to(device)
        #         target_jt_B = torch.from_numpy(data_B['jt'])#.to(device)
        #         target_global_pos_B = torch.from_numpy(data_B['global'])[:,:3]#.to(device)
        #         target_global_ori_B = torch.from_numpy(data_B['global'])[:,20*3:20*3+6]#.to(device)
        #         self.jt_A.append(target_jt_A)
        #         self.jt_B.append(target_jt_B)
        #         self.root_A.append(torch.concat([target_global_pos_A, target_global_ori_A], dim=1))
        #         self.root_B.append(torch.concat([target_global_pos_B, target_global_ori_B], dim=1))
        #         motion_length.append(target_jt_A.shape[0])
        # # target_jt = torch.cat(target_jt, dim=0).to(device)
        # # target_global = torch.cat(target_global, dim=0).to(device)
        # start_id = torch.zeros_like(target_length, dtype=torch.long)
        # start_id[1:] = torch.cumsum(target_length[:-1], dim=0)
        recycle_data_dict = joblib.load(recycle_data_path)
        retarget_data_dict = joblib.load(retarget_data_path)
        self.load_human = human_data_path is not None
        if human_data_path is not None:
            human_data_dict = joblib.load(human_data_path)
            # import pytorch_kinematics as pk
            # chain = pk.build_chain_from_urdf(open("/home/ubuntu/workspace/H1_RL/HST/legged_gym/resources/robots/h1/urdf/h1_add_hand_link_for_pk.urdf","rb").read())
            # human_node_names=['Pelvis', 
            # 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 
            # 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 
            # 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 
            # 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 
            # 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        for name, recycle_data in recycle_data_dict.items():
            if name not in retarget_data_dict.keys():
                continue
            # if data_pair is None:
            #     continue
            # name = data_pair['name']
            self.names.append(name)
        
            retarget_jt = torch.from_numpy(retarget_data_dict[name]['jt'])[:,:19]#.to(device)
            retarget_global_pos = torch.from_numpy(retarget_data_dict[name]['global'])[:,:3]#.to(device)
            retarget_global_ori = torch.from_numpy(retarget_data_dict[name]['global'])[:,h1_num_bodies*3:h1_num_bodies*3+6]#.to(device)
            retarget_root = torch.concat([retarget_global_pos, retarget_global_ori], dim=1)
                # breakpoint()
            recycle_jt = torch.from_numpy(recycle_data['jt'])[:,:19]#.to(device)
            recycle_global_pos = torch.from_numpy(recycle_data['global'])[:,:3]#.to(device)
            recycle_global_ori = torch.from_numpy(recycle_data['global'])[:,h1_num_bodies*3:h1_num_bodies*3+6]#.to(device)

            # ret = chain.forward_kinematics(target_jt_B)
            # look up the transform for a specific link
            # left_hand_link = ret['left_hand_link']
            # left_hand_tg = left_hand_link.get_matrix()[:,:3,3]
            # right_hand_link = ret['right_hand_link']
            # right_hand_tg = right_hand_link.get_matrix()[:,:3,3]
            # left_ankle_link = ret['left_ankle_link']
            # left_ankle_tg = left_ankle_link.get_matrix()[:,:3,3]
            # right_ankle_link = ret['right_ankle_link']
            # right_ankle_tg = right_ankle_link.get_matrix()[:,:3,3]
            # get transform matrix (1,4,4), then convert to separate position and unit quaternion

            if human_data_path is not None:
                human_jt = human_data_dict[name]['local_rotation'].reshape(-1,(human_num_bodies-1)*6)
                # target_jt_A[:,0] = 0
                human_root = human_data_dict[name]['root_transformation']
                self.jt_C.append(human_jt)
                self.root_C.append(human_root)
                # breakpoint()


            self.jt_A.append(retarget_jt)
            self.jt_B.append(recycle_jt)
            self.root_A.append(retarget_root)
            self.root_B.append(torch.concat([recycle_global_pos, recycle_global_ori], dim=1))
            motion_length.append(retarget_jt.shape[0])
            assert retarget_jt.shape[0] == recycle_jt.shape[0]

        self.jt_A = torch.cat(self.jt_A, dim=0)
        self.jt_B = torch.cat(self.jt_B, dim=0)
        self.root_A = torch.cat(self.root_A, dim=0)
        self.root_B = torch.cat(self.root_B, dim=0)
        self.jt_root_A = torch.concat([self.jt_A, self.root_A], dim=1).type(torch.float32)
        self.jt_root_B = torch.concat([self.jt_B, self.root_B], dim=1).type(torch.float32)
        if human_data_path is not None:
            self.jt_C = torch.cat(self.jt_C, dim=0)
            self.root_C = torch.cat(self.root_C, dim=0)
            self.jt_root_C = torch.concat([self.jt_C, self.root_C], dim=1).type(torch.float32)
            del self.jt_C, self.root_C
        # normalize
        self.mean_A = self.jt_root_A.mean(dim=0, keepdim=True)
        self.mean_B = self.jt_root_B.mean(dim=0, keepdim=True)
        self.std_A = self.jt_root_A.std(dim=0, keepdim=True)
        self.std_B = self.jt_root_B.std(dim=0, keepdim=True)
        if norm:
            self.jt_root_A = (self.jt_root_A - self.mean_A) / self.std_A
            self.jt_root_B = (self.jt_root_B - self.mean_B) / self.std_B
            
        
        # self.cov_xy = 0*(self.jt_root_B * self.jt_root_B).mean() + 0.5 #dim=0, keepdim=True
        self.cov_xy=None
        # breakpoint()
        del self.jt_A, self.jt_B, self.root_A, self.root_B
        self.motion_length = torch.tensor(motion_length, dtype=torch.long)
        self.length = len(motion_length)
        self.max_length = self.motion_length.max()
        # self.pad = torch.zeros((max_length, 19+3+6)).to(self.jt_A)
        self.start_id = torch.zeros_like(self.motion_length, dtype=torch.long)
        self.start_id[1:] = torch.cumsum(self.motion_length[:-1], dim=0)
                    


    def __getitem__(self, index):
        # index=index%16
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B
            A (tensor) - - an motion in the input jt_A and root_A
            B (tensor) - - its corresponding target motion
        """
        motion_length = self.motion_length[index]
        while motion_length < self.window_size:
            index += 1
            index %= self.length
            motion_length = self.motion_length[index]
        # jt_A = self.jt_A[self.start_id[index]:self.start_id[index]+motion_length]
        # jt_B = self.jt_B[self.start_id[index]:self.start_id[index]+motion_length]
        # root_A = self.root_A[self.start_id[index]:self.start_id[index]+motion_length]
        # root_B = self.root_B[self.start_id[index]:self.start_id[index]+motion_length]
        # jt_root_A = torch.concat([jt_A, root_A], dim=1)
        # jt_root_B = torch.concat([jt_B, root_B], dim=1)
        jt_root_A = self.jt_root_A[self.start_id[index]:self.start_id[index]+motion_length]
        jt_root_B = self.jt_root_B[self.start_id[index]:self.start_id[index]+motion_length]
        if self.load_human:
            jt_root_C = self.jt_root_C[self.start_id[index]:self.start_id[index]+motion_length]
        # if self.load_pose:
        #     # random choose a pose 
        #     pose_id = torch.randint(motion_length, (1,))
        #     A = jt_root_A[pose_id]
        #     B = jt_root_B[pose_id]
        #     if self.load_human:
        #         C = jt_root_C[pose_id]
        elif self.train:
            pose_id = torch.randint(motion_length - self.window_size+1, (1,))
            if self.overlap >= 0:
                # breakpoint()
                pose_id = pose_id//(self.window_size-self.overlap) * (self.window_size-self.overlap)
            # zero_pad_A = torch.zeros((self.max_length-motion_length, jt_root_A.shape[-1])).to(jt_root_A)
            # zero_pad_B = torch.zeros((self.max_length-motion_length, jt_root_B.shape[-1])).to(jt_root_A)
            # A = torch.concat([jt_root_A, zero_pad_A], dim=0)
            # B = torch.concat([jt_root_B, zero_pad_B], dim=0)
            A = jt_root_A[pose_id:pose_id+self.window_size]
            B = jt_root_B[pose_id:pose_id+self.window_size]
            if self.load_human:
                # zero_pad_C = torch.zeros((self.max_length-motion_length, jt_root_C.shape[-1])).to(jt_root_A)
                # C = torch.concat([jt_root_C, zero_pad_C], dim=0)
                C = jt_root_C[pose_id:pose_id+self.window_size]
        else:
            A = jt_root_A
            B = jt_root_B
            if self.load_human:
                C = jt_root_C

        if self.load_human:
            if self.train:
                return B, A, C
            return B, A, C, index#self.names[index]
        return B, A, index#self.names[index]

        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length


def get_cond(global_pos_offset, root_quat, left_hand_tg, right_hand_tg, left_ankle_tg, right_ankle_tg):
    if global_pos_offset is None:
        left_hand_global=left_hand_tg
        right_hand_global=right_hand_tg
        left_ankle_global=left_ankle_tg
        right_ankle_global=right_ankle_tg
    else:
        left_hand_global = quat_apply(root_quat, left_hand_tg) + global_pos_offset
        right_hand_global = quat_apply(root_quat, right_hand_tg) + global_pos_offset
        left_ankle_global = quat_apply(root_quat, left_ankle_tg) + global_pos_offset
        right_ankle_global = quat_apply(root_quat, right_ankle_tg) + global_pos_offset

    mean_pos_global =  (left_hand_global + right_hand_global + left_ankle_global + right_ankle_global)/4
    heading_quat_inv = calc_heading_quat_inv(root_quat)

    left_hand_tg = quat_apply(heading_quat_inv, left_hand_global - mean_pos_global)
    right_hand_tg = quat_apply(heading_quat_inv, right_hand_global - mean_pos_global)
    left_ankle_tg = quat_apply(heading_quat_inv, left_ankle_global - mean_pos_global)
    right_ankle_tg = quat_apply(heading_quat_inv, right_ankle_global - mean_pos_global)

    cond_jt = torch.concat([left_hand_tg, right_hand_tg, left_ankle_tg, right_ankle_tg, mean_pos_global], dim=1)
    cond_root = calc_heading_quat(root_quat)[...,2:]
    return cond_jt, cond_root

class MotionDataset(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, recycle_data_path, retarget_data_path, train=True, human_data_path = None, load_pose = False, norm = False, overlap=-1,window_size=24,mixed_data=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        # data_list_A = joblib.load(data_path_A)
        # data_list_B = joblib.load(data_path_B)
        self.jt_A = []
        self.jt_B = []
        self.jt_C = []
        self.root_A = []
        self.root_B = []
        self.root_C = []
        motion_length = []
        self.names = []
        self.load_pose = load_pose
        h1_num_bodies = 22
        human_num_bodies = 24
        self.window_size = window_size
        # self.overlap = overlap
        self.mixed_data = mixed_data
        self.train=train
        # start_id = []
        # current_id = 0
        # for (name, data_A), data_B in zip(data_list_A.items(), data_list_B):
        #     if data_B is not None:
        #         target_jt_A = torch.from_numpy(data_A['jt'])#.to(device)
        #         target_global_pos_A = torch.from_numpy(data_A['global'])[:,:3]#.to(device)
        #         target_global_ori_A = torch.from_numpy(data_A['global'])[:,20*3:20*3+6]#.to(device)
        #         target_jt_B = torch.from_numpy(data_B['jt'])#.to(device)
        #         target_global_pos_B = torch.from_numpy(data_B['global'])[:,:3]#.to(device)
        #         target_global_ori_B = torch.from_numpy(data_B['global'])[:,20*3:20*3+6]#.to(device)
        #         self.jt_A.append(target_jt_A)
        #         self.jt_B.append(target_jt_B)
        #         self.root_A.append(torch.concat([target_global_pos_A, target_global_ori_A], dim=1))
        #         self.root_B.append(torch.concat([target_global_pos_B, target_global_ori_B], dim=1))
        #         motion_length.append(target_jt_A.shape[0])
        # # target_jt = torch.cat(target_jt, dim=0).to(device)
        # # target_global = torch.cat(target_global, dim=0).to(device)
        # start_id = torch.zeros_like(target_length, dtype=torch.long)
        # start_id[1:] = torch.cumsum(target_length[:-1], dim=0)
        recycle_data_dict = joblib.load(recycle_data_path)
        retarget_data_dict = joblib.load(retarget_data_path)
        self.cond_train = human_data_path is not None and self.train
        self.cond_test = human_data_path is not None and not self.train
        if self.cond_train:
            import pytorch_kinematics as pk
            self.chain = pk.build_chain_from_urdf(open("./ddbm/h1/urdf/h1_add_hand_link_for_pk.urdf","rb").read())

        names_50 = joblib.load('/cephfs_yili/shared/xuehan/H1_RL/tracked_2165_0.5_names.pkl')

        if self.cond_test:
            human_data_dict = joblib.load(human_data_path)
            
        
        for name, recycle_data in recycle_data_dict.items():
            if name not in retarget_data_dict.keys() or name not in names_50:
                continue
            # if data_pair is None:
            #     continue
            # name = data_pair['name']
            self.names.append(name)
        
            retarget_jt = torch.from_numpy(retarget_data_dict[name]['jt'])[:,:19]#.to(device)
            retarget_global_pos = torch.from_numpy(retarget_data_dict[name]['global'])[:,:3]#.to(device)
            retarget_global_ori = torch.from_numpy(retarget_data_dict[name]['global'])[:,h1_num_bodies*3:h1_num_bodies*3+6]#.to(device)
            retarget_root = torch.concat([retarget_global_pos, retarget_global_ori], dim=1)
                # breakpoint()
            recycle_jt = torch.from_numpy(recycle_data['jt'])[:,:19]#.to(device)
            recycle_global_pos = torch.from_numpy(recycle_data['global'])[:,:3]#.to(device)
            recycle_global_ori = torch.from_numpy(recycle_data['global'])[:,h1_num_bodies*3:h1_num_bodies*3+6]#.to(device)
            # breakpoint()

            if self.cond_test:
                human_link = human_data_dict[name]['global_translation'].reshape(-1,(human_num_bodies),3)
                to_ground = human_link[...,-1].min()
                human_link[...,-1] = human_link[...,-1] - to_ground
                left_hand_tg= human_link[:,24-6]
                right_hand_tg= human_link[:,24-1]
                left_ankle_tg = human_link[:,3]
                right_ankle_tg = human_link[:,7]
                human_root = human_data_dict[name]['root_transformation']
                root_quat = vec6d_to_quat(human_root[:,-6:].reshape(-1,3,2))
                human_jt, human_root = get_cond(None, root_quat, left_hand_tg, right_hand_tg, left_ankle_tg, right_ankle_tg)
                # target_jt_A[:,0] = 0
                self.jt_C.append(human_jt)
                self.root_C.append(human_root)
            elif self.cond_train:
                # breakpoint()
                ret = self.chain.forward_kinematics(recycle_jt)
                left_hand_link = ret['left_hand_link']
                left_hand_tg = left_hand_link.get_matrix()[:,:3,3]
                right_hand_link = ret['right_hand_link']
                right_hand_tg = right_hand_link.get_matrix()[:,:3,3]
                left_ankle_link = ret['left_ankle_link']
                left_ankle_tg = left_ankle_link.get_matrix()[:,:3,3]
                right_ankle_link = ret['right_ankle_link']
                right_ankle_tg = right_ankle_link.get_matrix()[:,:3,3]
                # pelvis_link = ret['left_knee_link']
                # pelvis_tg = pelvis_link.get_matrix()[:,:3,3]
                root_quat = vec6d_to_quat(recycle_global_ori.reshape(-1,3,2))
                cond_jt, cond_root = get_cond(recycle_global_pos, root_quat, left_hand_tg, right_hand_tg, left_ankle_tg, right_ankle_tg)

                self.jt_C.append(cond_jt)
                self.root_C.append(cond_root)


            self.jt_A.append(retarget_jt)
            self.jt_B.append(recycle_jt)
            self.root_A.append(retarget_root)
            self.root_B.append(torch.concat([recycle_global_pos, recycle_global_ori], dim=1))
            motion_length.append(retarget_jt.shape[0])
            assert retarget_jt.shape[0] == recycle_jt.shape[0]

        self.jt_A = torch.cat(self.jt_A, dim=0)
        self.jt_B = torch.cat(self.jt_B, dim=0)
        self.root_A = torch.cat(self.root_A, dim=0)
        self.root_B = torch.cat(self.root_B, dim=0)
        self.jt_root_A = torch.concat([self.jt_A, self.root_A], dim=1).type(torch.float32)
        self.jt_root_B = torch.concat([self.jt_B, self.root_B], dim=1).type(torch.float32)
        if human_data_path is not None:
            self.jt_C = torch.cat(self.jt_C, dim=0)
            self.root_C = torch.cat(self.root_C, dim=0)
            self.jt_root_C = torch.concat([self.jt_C, self.root_C], dim=1).type(torch.float32)
            del self.jt_C, self.root_C
        # normalize
        self.mean_A = self.jt_root_A.mean(dim=0, keepdim=True)
        self.mean_B = self.jt_root_B.mean(dim=0, keepdim=True)
        self.std_A = self.jt_root_A.std(dim=0, keepdim=True)
        self.std_B = self.jt_root_B.std(dim=0, keepdim=True)
        if norm:
            self.jt_root_A = (self.jt_root_A - self.mean_A) / self.std_A
            self.jt_root_B = (self.jt_root_B - self.mean_B) / self.std_B
            
        
        # self.cov_xy = 0*(self.jt_root_B * self.jt_root_B).mean() + 0.5 #dim=0, keepdim=True
        self.cov_xy=None
        # breakpoint()
        del self.jt_A, self.jt_B, self.root_A, self.root_B
        self.motion_length = torch.tensor(motion_length, dtype=torch.long)
        self.length = len(motion_length)
        self.max_length = self.motion_length.max()
        # self.pad = torch.zeros((max_length, 19+3+6)).to(self.jt_A)
        self.start_id = torch.zeros_like(self.motion_length, dtype=torch.long)
        self.start_id[1:] = torch.cumsum(self.motion_length[:-1], dim=0)
                    


    def __getitem__(self, index):
        # index=index%16
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B
            A (tensor) - - an motion in the input jt_A and root_A
            B (tensor) - - its corresponding target motion
        """
        motion_length = self.motion_length[index]
        while motion_length < self.window_size:
            index += 1
            index %= self.length
            motion_length = self.motion_length[index]

        jt_root_A = self.jt_root_A[self.start_id[index]:self.start_id[index]+motion_length]
        jt_root_B = self.jt_root_B[self.start_id[index]:self.start_id[index]+motion_length]
        cond =  self.cond_train or self.cond_test
        if cond:
            jt_root_C = self.jt_root_C[self.start_id[index]:self.start_id[index]+motion_length]

        if self.train:
            pose_id = torch.randint(motion_length - self.window_size+1, (1,))

            A = jt_root_A[pose_id:pose_id+self.window_size]
            B = jt_root_B[pose_id:pose_id+self.window_size]
            if cond:
                C = jt_root_C[pose_id:pose_id+self.window_size]
                return B, A, C, index
            return B, A, index
        else:
            A = jt_root_A
            B = jt_root_B
            if cond:
                C = jt_root_C
                return B, A, C, index
            return B, A, index

        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length
