import argparse

from .karras_diffusion import KarrasDenoiser
# from .unet import UNetModel
# from .edm_unet import SongUNet
from .mrm import MRM
from .mrm import ContrastiveModel
import numpy as np

NUM_CLASSES = 1000


def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir

def cm_train_defaults():
    return dict(
        teacher_model_path="",
        teacher_dropout=0.1,
        training_mode="consistency_distillation",
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=600000,
        start_ema=0.0,
        start_scales=40,
        end_scales=40,
        distill_steps_per_iter=50000,
        loss_norm="lpips",
    )

def sample_defaults():
    return dict(
        generator="determ",
        clip_denoised=True,
        sampler="euler",
        s_churn=0.0,
        s_tmin=0.002,
        s_tmax=80,
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80.0,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0.,
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        unet_type='adm',
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        attention_type='flash',
        learn_sigma=False,
        condition_mode=None,
        pred_mode='ve',
        weight_schedule="karras",
        layers=8,
        # latent_dim=512,
        cond_mask_prob=1.0,
        emb_trans_dec=True,
        arch='trans_enc',
    )
    return res


def create_model_and_diffusion(
    image_size,
    in_channels,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    attention_type,
    condition_mode,
    pred_mode,
    weight_schedule,
    sigma_data=0.5,
    sigma_min=0.002,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    cov_xy=0.,
    unet_type='adm',
):
    model = create_model(
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        unet_type=unet_type,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        attention_type=attention_type,
        condition_mode=condition_mode,
    )
    diffusion = KarrasDenoiser(
        sigma_data=sigma_data,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        beta_d=beta_d,
        beta_min=beta_min,
        cov_xy=cov_xy,
        image_size=image_size,
        weight_schedule=weight_schedule,
        pred_mode=pred_mode
    )
    return model, diffusion


def create_model_and_diffusion_mrm(
    args, cov_xy = None,
):
    model = MRM(**get_model_args_mrm(args))


    if args.vae_checkpoint is not None:
        vae= ContrastiveModel(**get_model_args_mrm(args))
    else:
        vae = None
    diffusion = KarrasDenoiser(
        sigma_data=args.sigma_data,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        # beta_d=beta_d,
        # beta_min=beta_min,
        cov_xy=args.cov_xy if cov_xy is None else cov_xy,
        # image_size=image_size,
        weight_schedule=args.weight_schedule,
        pred_mode=args.pred_mode,
        # vae=vae,
    )
    return model, diffusion, vae


def get_model_args_mrm(args):

    # default args
    # clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = args.condition_mode
    # if hasattr(data.dataset, 'num_actions'):
    #     num_actions = data.dataset.num_actions
    # else:
    num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 28
    nfeats = 1

    # if args.dataset == 'humanml':
    #     data_rep = 'hml_vec'
    #     njoints = 263
    #     nfeats = 1
    # elif args.dataset == 'kit':
    #     data_rep = 'hml_vec'
    #     njoints = 251
    #     nfeats = 1
    args.use_latent = False # args.human_data_path is not None

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'use_fp16': args.use_fp16, 'use_latent': args.use_latent,
            'sigma_max': args.sigma_max, 'sigma_min': args.sigma_min,
            'latent_dim': args.num_channels, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': args.dropout, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec}#, 'dataset': args.dataset}

def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
