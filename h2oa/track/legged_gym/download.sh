exp_name=0809_8554recycle

# for dir in $(find /home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/rough_h1/ -type d -name "$exp_name"); do
mkdir -p /home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/rough_h1/$exp_name
scp -r xuehan@10.210.5.13:/home/xuehan/H1_RL/HST/legged_gym/logs/rough_h1/$exp_name/ /home/ubuntu/workspace/H1_RL/HST/legged_gym/logs/rough_h1/
# done