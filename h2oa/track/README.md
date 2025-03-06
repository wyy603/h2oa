# HumanPlus: Humanoid Shadowing and Imitation from Humans


#### Project Website: https://humanoid-ai.github.io/

This repository contains the updating implementation for the Humanoid Shadowing Transformer (HST) and the Humanoid Imitation Transformer (HIT), along with instructions for whole-body pose estimation and the associated hardware codebase.


## Humanoid Shadowing Transformer (HST)
Reinforcement learning in simulation is based on [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
#### Installation
Install IsaacGym v4 first from the [official source](https://developer.nvidia.com/isaac-gym). Place the isaacgym fold inside the HST folder.

    cd HST/rsl_rl && pip install -e . 
    cd HST/legged_gym && pip install -e .

#### Example Usages
To train HST: d467077c5e839616092d5f0761606c729cc6108c

CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name 1110_Truemc_8204dn --sim_device cuda:0 --rl_device cuda:0 --task h1_mc --headless  


 --resume --load_run 1105_mc_8204dn


CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name 1107_lctk_walk_dn --sim_device cuda:0 --rl_device cuda:0 --task h1_lctk  --headless  --resume --load_run 1028_lctk_walkforward_hipyaw_newmotion_dr_resume

CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name 1111_box_randstill --sim_device cuda:0 --rl_device cuda:0 --task h1_lctk  --headless --resume --load_run 1102_17box

CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name 1110_h2o_swingwalk --sim_device cuda:0 --rl_device cuda:0 --task h1_h2o  --headless

--resume --load_run 1029_lctk_still_dr_resume

--headless


--debug

CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name 1101_sim15_8204 --sim_device cuda:0 --rl_device cuda:0 --task h1_phc  --headless --resume --load_run 0831_11112retar_tarinit 

CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/train.py --run_name 1103_simhp8204 --sim_device cuda:0 --rl_device cuda:0 --task h1_hp  --headless 
CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/train.py --run_name 1101_simhpours8204 --sim_device cuda:0 --rl_device cuda:0 --task h1_phc  --headless --resume --load_run 0831_11112retar_tarinit 
CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name 1111_handup --sim_device cuda:0 --rl_device cuda:0 --task h1_lctk  --headless
  --resume --load_run 1028_lctk_walkforward_hipyaw_newmotion_dr_resume

--debug

 --resume --load_run 1017_s2r_8
--debug
 --debug
--resume --load_run 1013_s2r15_1234  --debug
--headless --resume --load_run 1013_s2r15_1234  
--debug

CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name 1025_simpop8198 --sim_device cuda:0 --rl_device cuda:0 --task h1_phc  --headless --resume --load_run 0831_11112retar_tarinit 
CUDA_VISIBLE_DEVICES=2 python legged_gym/scripts/train.py --run_name 0927_ezdn_dr --sim_device cuda:0 --rl_device cuda:0 --task h1_phc  --headless 
--resume --load_run 0831_11112retar_tarinit 
    --checkpoint 30000
    --debug
    

CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name test --sim_device cuda:0 --rl_device cuda:0 --task h1_mc --headless  

    --resume --load_run 0730_mlp_phc_global_5000motion_rootval

    
    NOTE!!!!!!!!!! mlp rsl_rl
    CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --run_name mlp_walk_phc_local_5motion_0726 --sim_device cuda:0 --rl_device cuda:0 --task h1_phc --headless 

To play a trained policy:


python legged_gym/scripts/play.py --load_run 1106_h2o_box_dr --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_h2o
    --headles

python legged_gym/scripts/play.py --load_run 1111_box_randstill --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_lctk

python legged_gym/scripts/play.py --load_run 1108_mcsr_hand --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_mcsr

    --headless test_hp

test:


CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/export.py --load_run 1110_hand_randstill --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_lctk --headless 

CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/test.py --load_run 1027_lctk_cvel_still_dr_resume_still --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_lctk --headless 

CUDA_VISIBLE_DEVICES=2 python legged_gym/scripts/get_recycle_data.py --load_run 1019_sim8198 --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_phc --headless 

CUDA_VISIBLE_DEVICES=1 python legged_gym/scripts/get_recycle_data.py --load_run 1110_Truemc_8204dn --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_mc --headless 

CUDA_VISIBLE_DEVICES=7 python legged_gym/scripts/get_recycle_data.py --load_run 1105_mc_8204dn --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_mc --headless 



CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/get_fail_data.py --load_run 0730_mlp_phc_global_5000motion_norootval --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --task h1_phc --headless 

conda activate hst
cd H1_RL/HST/legged_gym


0730_mlp_phc_global_5000motion_norootval

filename=h1_phc_config.py

scp xuehan@10.210.5.13:/home/xuehan/H1_RL/HST/legged_gym/logs/h1-rl/hst_walk_phc_global_novel_0724/model_22000.pt /home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/rough_h1/hst_walk_phc_global_novel_0724


mlp_walk_phc_local_jlpp_debug_0726
mlp_walk_phc_local_debug_0726
mlp_walk_phc_local_energy_0726
mlp_walk_phc_local_energy_curriculum_0726

filename=h1_phc_config.py
filename=model_30000.pt
exp_name=0919_2165_denoise
mkdir /home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/rough_h1/$exp_name/
scp xuehan@10.210.5.15:/home/xuehan/H1_RL/HST/legged_gym/logs/rough_h1/$exp_name/$filename /home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/rough_h1/$exp_name/

scp xuehan@10.210.5.15:/cephfs_yili/shared/xuehan/H1_RL/amass_train_13912.pkl /home/ubuntu/data/PHC/
scp xuehan@10.210.5.15:/cephfs_yili/shared/xuehan/H1_RL/retarget_13911_amass_train_13912.pkl /home/ubuntu/data/PHC/


## Humanoid Imitation Transformer (HIT)
Imitation learning in the real world is based on [ACT repo](https://github.com/tonyzhaozh/act) and [Mobile ALOHA repo](https://github.com/MarkFzp/act-plus-plus).
#### Installation
    conda create -n HIT python=3.8.10
    conda activate HIT
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    pip install getkey
    pip install wandb
    pip install chardet
    pip install h5py_cache
    cd HIT/detr && pip install -e .
#### Example Usages
Collect your own data or download our dataset from [here](https://drive.google.com/drive/folders/1i3eGTd9Nl_tSieoE0grxuKqUAumBr2EV?usp=drive_link) and place it in the HIT folder.
gdown https://drive.google.com/uc?id=1cBca7CZ19UVSE_cZPulMRypgE8L-z7QV -O  /cephfs_yili/backup/xuehan/dataset/humanplus/
https://drive-data-export.usercontent.google.com/download/frpfck3rkm6vhfi9f1dk1o7ftjbk6593/ntpq4a2lkmh6l2kel5vpmo21k0hqraju/1719045000000/77441ad9-c067-40c5-bb90-ed53a0ade2ef/102624654008619659056/ADt3v-NYX8MyqMUDEldicfZ_LeFeXBFqkVGH1JQVUmi1cy0KxfojyTYfGPdtsM7x-6FAD-HeQ5hre9VOKJesbOmrXEDcCamkjUT1-GHbzHCZ4cD8Ch0D4zCHjUbz9APXc_4w955vWMU6aNBcJriXMXASJTW3uSBAzpQpi5QkCN0wM3DbZPyHkjClchM4HZsM2nqi8jJkelyNIGaeweZV0mY50E23sDXvtDcqegx9BHCJWkfBvhwTDQl-60fPrVuWmeFxi3s8UtfHUMO1jBQo_5ogwiEzZCcJ04j427VO8vCEL-8DWxNJbMcs17VRzY-8SIhGnxd_yrG6?j=77441ad9-c067-40c5-bb90-ed53a0ade2ef&user=191826100437&i=0&authuser=0
https://drive.google.com/drive/folders/1cBca7CZ19UVSE_cZPulMRypgE8L-z7QV?usp=drive_link

To set up a new terminal, run:

    conda activate HIT
    cd HIT

To train HIT:

    # Fold Clothes task
    python imitate_episodes_h1_train.py --task_name data_fold_clothes --ckpt_dir fold_clothes/ --policy_class HIT --chunk_size 50 --hidden_dim 512 --batch_size 48 --dim_feedforward 512 --lr 1e-5 --seed 0 --num_steps 100000 --eval_every 100000 --validate_every 1000 --save_every 10000 --no_encoder --backbone resnet18 --same_backbones --use_pos_embd_image 1 --use_pos_embd_action 1 --dec_layers 6 --gpu_id 0 --feature_loss_weight 0.005 --use_mask --data_aug


## Pose Estimation
For body pose estimation, please refer to [WHAM](https://github.com/yohanshin/WHAM). 
For hand pose estimation, please refer to [HaMeR](https://github.com/geopavlakos/hamer). 


## Hardware Codebase
Hardware codebase is based on [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2).

#### Installation

install [unitree_sdk](https://github.com/unitreerobotics/unitree_sdk2)

install [unitree_ros2](https://support.unitree.com/home/en/developer/ROS2_service)

    conda create -n lowlevel python=3.8
    conda activate lowlevel

install [nvidia-jetpack](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-461/install-jetpack/index.html)

install torch==1.11.0 and torchvision==0.12.0:  
please refer to the following links:   
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

#### Example Usages
Put your trained policy in the `hardware-script/ckpt` folder and rename it to `policy.pt`

    conda activate lowlevel
    cd hardware-script
    python hardware_whole_body.py --task_name stand
