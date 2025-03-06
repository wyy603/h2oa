# training
BS=200
MICRO=200
NGPU=1
SAVE_ITER=1000


# diff
PRED=$2
SIGMA_MAX=1
SIGMA_MIN=0.002
SIGMA_DATA=0.5
COV_XY=0.
SAMPLER=real-uniform # real-uniform??

# data
# RETARGET_DATA_PATH='/home/ubuntu/data/PHC/retarget_64_amass_train_13912.pkl'
# RETARGET_DATA_PATH='/cephfs_yili/shared/xuehan/H1_RL/retarget_fail2464_amass_train_13912.pkl'
RETARGET_DATA_PATH='/cephfs_yili/shared/xuehan/H1_RL/retarget_13911_amass_train_13912.pkl'
# RECYCLE_DATA_PATH='/home/ubuntu/data/PHC/tracked_64_0.5_0831_11112retar_tarinit_retarget_64_amass_train_13912.pkl'
# RECYCLE_DATA_PATH='/cephfs_yili/shared/xuehan/H1_RL/retarget_fail2464_amass_train_13912.pkl'
RECYCLE_DATA_PATH='/cephfs_yili/shared/xuehan/H1_RL/retarget_13911_amass_train_13912.pkl'
# RECYCLE_DATA_PATH='/cephfs_yili/shared/xuehan/H1_RL/tracked_1949_0.25_0831_11112retar_tarinit_retarget_13911_amass_train_13912.pkl'
# HUMAN_DATA_PATH='/home/ubuntu/data/PHC/human_translation_6761_amass_isaac_train_0.pkl'
HUMAN_DATA_PATH='/cephfs_yili/shared/xuehan/H1_RL/human_13912_amass_train_13912.pkl'
NORMALIZE=False
ONLY_POSE=False
OVERLAP=8
WINDOW_SIZE=24
MIXED_DATA=False

# network
# ARCH='trans_enc' # 'debug'
ARCH='trans_enc'
ATTN=32,16,8
USE_16FP=False # True
ATTN_TYPE=flash

NUM_CH=256
# NUM_RES_BLOCKS=2
NUM_RES_BLOCKS=3

# extra
DATASET_NAME=$1
EXP="${DATASET_NAME}_${NUM_CH}d"
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    # SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "ve_simple" ]]; then
    EXP+="_ve_simple"
    COND=concat
elif  [[ $PRED == "vp_simple" ]]; then
    EXP+="_vp_simple"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    # SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi
EXP+="_${SIGMA_MAX}S"


