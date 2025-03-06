"""
Train a diffusion model on images.
"""

import argparse

from ddbm import logger , dist_util
from datasets import load_data_motion
from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion_mrm,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir
)
from ddbm.train_util import TrainLoop
import torch
import torch.distributed as dist

from pathlib import Path

import wandb
import numpy as np

from glob import glob
import os
from datasets.augment import AugmentPipe
def main(args):

    # if args.human_data_path is not None:
    #     args.exp += '_latent'
    # else:
    #     args.exp += '_raw'

    # if args.load_pose:
    #     args.exp += '_pose'
    # else:
    #     args.exp += '_motion'

    # if args.normalize:
    #     args.exp += '_norm'
    # else:
    #     args.exp += '_no_norm'



    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    # copy ./args.sh to workdir/args.sh
    import shutil
    shutil.copy('./args.sh', f'{workdir}/args.sh')

    # with open('./args.sh', 'r') as f:
    #     with open(f'{workdir}/args.sh', 'w') as f2:
    #         f2.write(f.read())
        
    
    dist_util.setup_dist()
    logger.configure(dir=workdir)
    # logger.logkv('exp',args.exp)
    

    # data_image_size = args.image_size
    

    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt
                if dist.get_rank() == 0:
                    logger.log('Resuming from checkpoint: ', max_ckpt)

    if args.human_data_path is not None:
        vae_workdir = f'workdir/enc_{args.num_channels}d_new'
        vae_ckpts = list(glob(f'{vae_workdir}/*model*[0-9].*'))
        if len(vae_ckpts) > 0:
            max_ckpt = max(vae_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
            if os.path.exists(max_ckpt):
                args.vae_checkpoint = max_ckpt
                if dist.get_rank() == 0:
                    logger.log('Resuming from vae_checkpoint: ', max_ckpt)


    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size
        
    if dist.get_rank() == 0:
        logger.log("creating data loader...")

    data, test_data, cov_xy = load_data_motion(
        recycle_data_path=args.recycle_data_path,
        retarget_data_path=args.retarget_data_path,
        # data_path_B=args.data_path_B,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        human_data_path=args.human_data_path,
        train=True,
        load_pose = args.load_pose,
        norm = args.normalize,
        overlap=args.overlap,
        window_size=args.window_size,
        mixed_data=args.mixed_data,
    )

    logger.log("creating model and diffusion...")
    model, diffusion, vae = create_model_and_diffusion_mrm(
        args, cov_xy
    )
    model.to(dist_util.dev())
    if vae is not None:
        vae.load_state_dict(
            torch.load(args.vae_checkpoint, map_location=dist_util.dev()),
            )
        vae.to(dist_util.dev())
        vae.eval()
        diffusion.vae = vae
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + '_resume'
        wandb.init(project="bridge", 
                   group=args.exp,
                   name=name, 
                    entity="axian",
                   config=vars(args), 
                   mode='online' if not args.debug else 'disabled')
    # if dist.get_rank() == 0:
        wandb.watch(model, log='all')

    

    
    if args.use_augment:
        augment = AugmentPipe(
                p=0.12,xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
            )
    else:
        augment = None
        
    # breakpoint()
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        augment_pipe=augment,
        mixed_data=args.mixed_data,
        **sample_defaults()
    ).run_loop()


def create_argparser():
    defaults = dict(
        # data_dir="",
        # dataset='edges2handbags',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=16,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=500,
        save_interval=10000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=2,
        use_augment=False,
        recycle_data_path=None,
        retarget_data_path=None,
        human_data_path=None,
        load_pose=False,
        normalize=False,
        vae_checkpoint=None,
        overlap=-1,
        window_size=24,
        mixed_data=False,
        # data_path='/cephfs_yili/shared/xuehan/H1_RL/recycle_8554.pkl',
        # data_path_B='/home/ubuntu/data/PHC/recycle_data_500.pkl',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    print('----------')
    args = create_argparser().parse_args()
    main(args)
