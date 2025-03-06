## installation

```
conda install -y python==3.8

cd ./h2oa/diffusion
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja
conda install -y -c conda-forge mpi4py openmpi
pip install -e .

cd ../../h2oa/track/isaacgym/python
conda install -y numpy=1.23
pip install -e .
cd ../../rsl_rl
pip install -e .
cd ../legged_gym
pip install -e .
pip install pydelatin wandb pytorch_kinematics pytorch3d ipdb onnx

cd ../../retarget/poselib
pip install -e .
cd ../SMPLSim
pip install -e .

cd ../../../
pip install -e .
```

# track

- train.py / train_h1_mc.sh: 训练用，train_h1_mc.sh 给出了一个可以用的命令行，环境是 h1_mc。与它有关的配置文件在 envs/h1_mc.py 和 envs/h1_mc_config.py

- replay.py: 可视化一个 h1 motion，需要在程序中更改 motoin_name 和 motion_data_path

- play.py / play_h1_mc.sh