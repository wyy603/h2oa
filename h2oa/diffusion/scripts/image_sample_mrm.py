"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torchvision.utils as vutils
import torch.distributed as dist

from ddbm import dist_util, logger
from ddbm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion_mrm,
    add_dict_to_argparser,
    args_to_dict,
)
from ddbm.random_util import get_generator
from ddbm.karras_diffusion import karras_sample, karras_sample_overlap, forward_sample

from datasets import load_data_motion

from pathlib import Path

from PIL import Image
# def get_workdir(exp):
#     workdir = f'./workdir/{exp}'
#     return workdir

def split_into_windows(length, window_size, overlap_size):
    assert window_size > overlap_size
    assert length >= window_size
    if length == window_size:
        return [(0, 0)]
    overlap = []
    start = 0
    end = 0
    # for i in range(0, length - overlap, window_size - overlap):
    #     start = min(i, length - window_size)
    #     windows.append((start, end))
    #     end = start + window_size
    while end < (length - window_size):
        overlap.append((start, end))
        end = start + window_size
        start += window_size - overlap_size
    assert overlap[-1][-1] <= (length - window_size)
    overlap.append((length - window_size, end))
    return overlap

def main():
    args = create_argparser().parse_args()

    workdir = os.path.dirname(args.model_path)

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.split("_")
    step = int(split[-1].split(".")[0])
    sample_dir = Path(workdir)/f'sample_{step}/w={args.guidance}_churn={args.churn_step_ratio}_{args.pred_mode}'
    dist_util.setup_dist()
    if dist.get_rank() == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=workdir)


    logger.log("creating model and diffusion...")
    model, diffusion, vae = create_model_and_diffusion_mrm(
        args, 
    )
    if vae is not None:
        vae.load_state_dict(
            th.load(args.vae_checkpoint, map_location=dist_util.dev()),
            )
        vae.to(dist_util.dev())
        vae.eval()
        diffusion.vae = vae
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model = model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    

    all_images = []
    all_indices = []
    

    all_dataloaders = load_data_motion(
        recycle_data_path=args.recycle_data_path,
        retarget_data_path=args.retarget_data_path,
        # data_path=args.data_path,
        # data_path_B=args.data_path_B,
        deterministic = True,
        batch_size=args.batch_size,
        train = False,
        num_workers=args.num_workers,
        seed=args.seed,
        human_data_path=args.human_data_path,
        load_pose = args.load_pose,
        norm = args.normalize,
        overlap=args.overlap,
        # window_size=args.window_size,
    )
    # if args.split =='train':
    dataloader = all_dataloaders[1]
    # elif args.split == 'test':
    #     dataloader = all_dataloaders[2]
    # else:
    #     raise NotImplementedError
    args.num_samples = len(dataloader.dataset)
    names = dataloader.dataset.names

    motion_pkls = {}
    denoised_error_all = []
    x_error_all = []
    
    for i, data in enumerate(dataloader):
        
        x0 = data[0].to(dist_util.dev())
        
        # x0 = x0_image.to(dist_util.dev()) * 2 -1
        
        y0 = data[1].to(dist_util.dev())
        cond = data[2].to(dist_util.dev())
        
        # y0 = y0_image.to(dist_util.dev()) * 2 - 1
        model_kwargs = {'xT': y0, 'cond': cond}

        bs = x0.shape[0]
        length, window_size, overlap = x0.shape[1], args.window_size, args.overlap
        overlap_pairs = split_into_windows(length, window_size, overlap)
        # breakpoint()
         
        sample, path, nfe, denoised_error, x_error = karras_sample_overlap(
            diffusion,
            model,
            y0,
            x0,
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            # clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=diffusion.sigma_min,
            sigma_max=diffusion.sigma_max,
            churn_step_ratio=args.churn_step_ratio,
            rho=args.rho,
            guidance=args.guidance,
            x_overlap=overlap_pairs,
        )  


        # sample, denoised_error, x_error = [], [], []
        # for start_i, overlap_i in overlap_pairs:
        #     if start_i < overlap_i:
        #         x_overlap = sample_last[:, start_i-overlap_i:]
        #     else:
        #         x_overlap = None
        #     sample_last, _, _, denoised_error_last, x_error_last = karras_sample_overlap(
        #         diffusion,
        #         model,
        #         y0[:,start_i:start_i+window_size],
        #         x0[:,start_i:start_i+window_size],
        #         steps=args.steps,
        #         model_kwargs=model_kwargs,
        #         device=dist_util.dev(),
        #         # clip_denoised=args.clip_denoised,
        #         sampler=args.sampler,
        #         sigma_min=diffusion.sigma_min,
        #         sigma_max=diffusion.sigma_max,
        #         churn_step_ratio=args.churn_step_ratio,
        #         rho=args.rho,
        #         guidance=args.guidance,
        #         x_overlap=x_overlap,
        #     )   
        #     sample.append(sample_last[:,overlap_i-start_i:window_size])
        #     denoised_error.extend(denoised_error_last)
        #     x_error.extend(x_error_last)
        # sample = th.cat(sample,dim=1)
        


        
        denoised_error_all.append(th.cat(denoised_error))
        x_error_all.append(th.cat(x_error))
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        # sample = sample.contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        gathered_xs = [th.zeros_like(x0) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_xs, x0)
        gathered_ys = [th.zeros_like(y0) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_ys, y0)

        index = data[3].to(dist_util.dev())
        
        if index is not None:
            gathered_index = [0 for _ in range(dist.get_world_size())]
            gathered_index = [th.zeros_like(index) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_index, index)
            # gathered_index = th.cat(gathered_index)
            # gathered_samples = gathered_samples[th.argsort(gathered_index)]
        # else:
        # gathered_samples = th.cat(gathered_samples)
        # gathered_xs = th.cat(gathered_xs)
        # gathered_ys = th.cat(gathered_ys)
        for samples, ids in zip(gathered_samples, gathered_index):
            for i in range(bs):
                name = names[ids[i].item()]
                motion_pkls[name] = {
                    'jt': samples[i][...,:19].detach().cpu().numpy(),
                    'root': samples[i][...,19:].detach().cpu().numpy(),
                }
                # breakpoint()
                # motion_pkls.append(motion_pkl)
            

        # num_display = min(32, sample.shape[0])
        # breakpoint()
        # if i == 0 and dist.get_rank() == 0:
        #     vutils.save_image(sample.permute(0,3,1,2)[:num_display].float(), f'{sample_dir}/sample_{i}.png', normalize=True,  nrow=int(np.sqrt(num_display)))
        #     if x0 is not None:
        #         vutils.save_image(x0_image[:num_display], f'{sample_dir}/x_{i}.png',nrow=int(np.sqrt(num_display)))
        #     vutils.save_image(y0_image[:num_display]/2+0.5, f'{sample_dir}/y_{i}.png',nrow=int(np.sqrt(num_display)))
            
            
        # gathered_samples = th.stack([gathered_samples,gathered_xs,gathered_ys],dim=1)
        # all_images.append(gathered_samples.detach().cpu().numpy())
        # all_indices.append(gathered_index.detach().cpu().numpy())

        
        
    # logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")
        

    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[:args.num_samples]
    denoised_error_all = th.stack(denoised_error_all)
    x_error_all = th.stack(x_error_all)
    denoised_error_mean = denoised_error_all.mean(dim=0)
    x_error_mean = x_error_all.mean(dim=0)
    # plot
    import matplotlib.pyplot as plt
    plt.plot(denoised_error_mean.cpu().numpy())
    plt.plot(x_error_mean.cpu().numpy())
    # saving
    if dist.get_rank() == 0:
        plt.savefig(f"{sample_dir}/error_{x_error_mean.cpu().numpy()[-1]}.png")

    logger.log(f"denoised error: {denoised_error_mean}")
    logger.log(f"x error: {x_error_mean}")
    if dist.get_rank() == 0:
        # shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        import joblib
        # change the first word of args.recycle_data_path before _ to denoise_jtroot
        # prefix = args.retarget_data_path.split("/")[-1].split("_")[0]
        # out_path=args.retarget_data_path.replace(prefix,"denoise_jtroot")
        out_file=f"dn_jtroot_{len(motion_pkls)}_{args.exp[:7]}_o{args.overlap}_s{step}_" + os.path.basename(args.retarget_data_path)
        out_path=os.path.join(os.path.dirname(args.retarget_data_path), out_file)
        # out_path=f"/home/ubuntu/data/PHC/{filename}"
        logger.log(f"saving to {out_path}")
        # breakpoint()
        joblib.dump(motion_pkls, out_path)
        # np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="", ## only used in bridge
        dataset='edges2handbags',
        # clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split='train',
        churn_step_ratio=0.,
        rho=7.0,
        steps=40,
        model_path="",
        exp="",
        seed=42,
        ts="",
        upscale=False,
        num_workers=2,
        guidance=1.,
        # data_path=None,
        recycle_data_path=None,
        retarget_data_path=None,
        human_data_path=None,
        load_pose=False,
        normalize=False,
        vae_checkpoint=None,
        overlap=-1,
        window_size=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
