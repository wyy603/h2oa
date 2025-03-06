

DATASET_NAME=$1
PRED=$2
MODEL_PATH=$3
CHURN_STEP_RATIO=$4
GUIDANCE=$5
SPLIT=$6


source ./args.sh $DATASET_NAME $PRED 


N=10
GEN_SAMPLER=heun
BS=1
NGPU=1

CUDA_VISIBLE_DEVICES=1 mpiexec -n $NGPU python scripts/image_sample_mrm.py --exp=$EXP \
    --batch_size $BS --churn_step_ratio $CHURN_STEP_RATIO --steps $N --sampler $GEN_SAMPLER \
    --model_path $MODEL_PATH --attention_resolutions $ATTN  --class_cond False --pred_mode $PRED \
    ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
    ${COND:+ --condition_mode="${COND}"} --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
    --dropout 0.1 --num_channels $NUM_CH --num_head_channels 64 --num_res_blocks $NUM_RES_BLOCKS \
    --resblock_updown True --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --use_scale_shift_norm True \
    --weight_schedule bridge_karras \
    --rho 7 --upscale=False ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
    ${UNET:+ --unet_type="${UNET}"} ${SPLIT:+ --split="${SPLIT}"} ${GUIDANCE:+ --guidance="${GUIDANCE}"} \
      --recycle_data_path $RECYCLE_DATA_PATH --retarget_data_path $RETARGET_DATA_PATH ${HUMAN_DATA_PATH:+ --human_data_path="${HUMAN_DATA_PATH}"} \
      --load_pose=$ONLY_POSE --arch=$ARCH --normalize=$NORMALIZE \
      --overlap=$OVERLAP --window_size=$WINDOW_SIZE
 

