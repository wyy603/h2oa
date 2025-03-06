"""
Based on: https://github.com/crowsonkb/k-diffusion
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# from piq import LPIPS


from .nn import mean_flat, append_dims, append_zero

from functools import partial


def vp_logsnr(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return - th.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)
    
def vp_logs(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80,
        sigma_min=0.002,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0., # 0 for uncorrelated, sigma_data**2 / 2 for  C_skip=1/2 at sigma_max
        rho=7.0,
        image_size=64,
        weight_schedule="karras",
        pred_mode='both',
        loss_norm="lpips",
        vae=None,
    ):
        self.sigma_data = sigma_data
        
        self.sigma_max = sigma_max 
        self.sigma_min = sigma_min 

        self.beta_d = beta_d
        self.beta_min = beta_min
        

        self.sigma_data_end = self.sigma_data
        self.cov_xy = cov_xy
            
        self.c = 1

        self.weight_schedule = weight_schedule
        self.pred_mode = pred_mode
        self.loss_norm = loss_norm
        # if loss_norm == "lpips":
        #     self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = 100#40
        self.vae = vae


    def get_snr(self, sigmas):
        if self.pred_mode.startswith('vp'):
            return vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas


    def get_weightings(self, sigma):
        snrs = self.get_snr(sigma)
        
        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            weightings = snrs + 1.0 / self.sigma_data**2
        elif self.weight_schedule.startswith("bridge_karras"):
            if self.pred_mode == 've':
                # sigma = append_dims(sigma, self.cov_xy.ndim)
                A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c**2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
                weightings = A / ((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )
            
            elif self.pred_mode == 'vp':
                
                logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
                logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
                logs_T = vp_logs(1, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -th.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

                # a_t = append_dims(a_t, self.cov_xy.ndim)
                # b_t = append_dims(b_t, self.cov_xy.ndim)
                # c_t = append_dims(c_t, self.cov_xy.ndim)

                A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
                weightings = A / (a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * c_t )
                
            elif self.pred_mode == 'vp_simple' or  self.pred_mode == 've_simple':

                weightings = th.ones_like(snrs)
        elif self.weight_schedule == "truncated-snr":
            weightings = th.clamp(snrs, min=1.0)
        elif self.weight_schedule == "uniform":
            weightings = th.ones_like(snrs)
        else:
            raise NotImplementedError()

        return weightings


    def get_bridge_scalings(self, sigma):
        if self.pred_mode == 've':
            # sigma = append_dims(sigma, self.cov_xy.ndim)
            A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c **2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
            c_in = 1 / (A) ** 0.5
            c_skip = ((1 - sigma**2 / self.sigma_max**2) * self.sigma_data**2 + sigma**2 / self.sigma_max**2 * self.cov_xy)/ A
            c_out =((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )**0.5 * c_in
            # breakpoint()
            return c_skip, c_out, c_in
        
    
        elif self.pred_mode == 'vp':

            logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
            logs_T = vp_logs(1, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -th.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

            # breakpoint()
            # a_t = append_dims(a_t, self.cov_xy.ndim)
            # b_t = append_dims(b_t, self.cov_xy.ndim)
            # c_t = append_dims(c_t, self.cov_xy.ndim)
            A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
            
            
            c_in = 1 / (A) ** 0.5
            c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy)/ A
            c_out =(a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * c_t )**0.5 * c_in
            return c_skip, c_out, c_in
            
    
        elif self.pred_mode == 've_simple' or self.pred_mode == 'vp_simple':
            
            c_in = th.ones_like(sigma)
            c_out = th.ones_like(sigma) 
            c_skip = th.zeros_like(sigma)
            return c_skip, c_out, c_in
        

    def training_bridge_losses(self, model, x0, sigmas, model_kwargs=None, noise=None, vae=None, train_diffusion=True):
        
        # breakpoint()
        assert model_kwargs is not None
        xT = model_kwargs['xT']
        sigmas =th.minimum(sigmas, th.ones_like(sigmas, dtype=sigmas.dtype)* self.sigma_max)
        # sigmas
        # sigmas=th.linspace(0, self.sigma_max, sigmas.shape[0],dtype=sigmas.dtype, device=sigmas.device)
        terms = {}

        dims = x0.ndim
        bs = x0.shape[0]
        if self.vae is not None:
            # breakpoint()
            with th.no_grad():
                L_A = self.vae.enc_A(xT)
                L_B = self.vae.enc_B(x0)
        else:
            L_A = xT
            L_B = x0
        model_kwargs['xT'] = L_A
        if noise is None:
            noise = th.randn_like(L_A) 
        def bridge_sample(x0, xT, t):
            t = append_dims(t, dims)
            # std_t = th.sqrt(t)* th.sqrt(1 - t / self.sigma_max)
            if self.pred_mode.startswith('ve'):
                std_t = t* th.sqrt(1 - t**2 / self.sigma_max**2)
                mu_t= t**2 / self.sigma_max**2 * xT + (1 - t**2 / self.sigma_max**2) * x0
                samples = (mu_t +  std_t * noise )
            elif self.pred_mode.startswith('vp'):
                logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
                logs_t = vp_logs(t, self.beta_d, self.beta_min)
                logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                std_t = (-th.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
                
                samples= a_t * xT + b_t * x0 + std_t * noise
            # breakpoint()
                
                
            return samples
        
        # breakpoint()
        # L_A = L_B
        x_t = bridge_sample(L_B, L_A, sigmas)
        # breakpoint()
        # x_t = L_A
        # sigmas=th.ones_like(sigmas, dtype=sigmas.dtype)* self.sigma_max
        

        terms["loss"] = 0
        # if train_diffusion:
        model_output, denoised = self.denoise(model, x_t, sigmas,  model_kwargs)
        if self.vae is not None:
            x0_denoised = self.vae.dec_B(denoised)
        else:
            x0_denoised = denoised

        weights = self.get_weightings(sigmas.type(x0.dtype))
        
        weights =  append_dims((weights), dims)
        # terms["xs_mse"] = mean_flat((x0_denoised - x0) ** 2)
        # terms["mse"] = mean_flat(weights * (x0_denoised - x0) ** 2) / th.std(L_B, dim=-1, keepdim = True)
        terms["mse/xs_loss"] = mean_flat((denoised - L_B) ** 2)
        terms["mse/dec_loss"] = mean_flat((x0_denoised - x0) ** 2)
        # terms["mse/loss"] = mean_flat(weights * (denoised - L_B) ** 2)
        terms["loss"] += mean_flat(weights * (denoised - L_B) ** 2)
        if "vb" in terms:
            terms["loss"] += terms["vb"]
        # else:
        # breakpoint()
        return terms
    


    def denoise(self, model, x_t, sigmas ,model_kwargs):
        # breakpoint()
        c_skip, c_out, c_in = [ # BUG!!!!!!!!! check the shape
            append_dims(x, x_t.ndim) for x in self.get_bridge_scalings(sigmas)
        ]
               
        # breakpoint()
        # rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        norm_sigmas = (sigmas - self.sigma_min) / (self.sigma_max - self.sigma_min)
        # norm_sigmas = sigmas
        model_output = model(c_in * x_t, norm_sigmas, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        # breakpoint()
        return model_output, denoised

def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    clip_denoised=False,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.,
    guidance=1,
):
    assert sampler in ["heun", ], 'only heun sampler is supported currently'
    
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)


    sample_fn = {
        "heun": partial(sample_heun, beta_d=diffusion.beta_d, beta_min=diffusion.beta_min),
    }[sampler]

    sampler_args = dict(
            pred_mode=diffusion.pred_mode, churn_step_ratio=churn_step_ratio, sigma_max=sigma_max
        )
    def denoiser(x_t, sigma, model_kwargs={}):
        _, denoised = diffusion.denoise(model, x_t, sigma, model_kwargs)
        
        # if clip_denoised:
        #     denoised = denoised.clamp(-1, 1)
                
        return denoised
    
    if diffusion.vae is not None:
        # breakpoint()
        with th.no_grad():
            L_A = diffusion.vae.enc_A(x_T)
            L_B_gt = diffusion.vae.enc_B(x_0)
    else:
        L_A = x_T
        L_B_gt = x_0

    # L_A = model.enc_A(x_T)
    # L_B_gt = model.enc_B(x_0)
    # breakpoint()
    L_B, path, nfe, denoised_error, x_error = sample_fn(
        denoiser,
        L_A,
        L_B_gt,
        sigmas,
        progress=progress,
        callback=callback,
        guidance=guidance,
        **sampler_args,
    )
    # breakpoint()
    print('nfe:', nfe)
    # x_0 = model.dec_B(L_B)
    if diffusion.vae is not None:
        # breakpoint()
        with th.no_grad():
            x_0 = diffusion.vae.dec_B(L_B)
    else:
        x_0 = L_B

    return x_0, path, nfe, denoised_error, x_error
    # return x_0.clamp(-1, 1), [x.clamp(-1, 1) for x in path], nfe

def karras_sample_overlap(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    clip_denoised=False,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.,
    guidance=1,
    x_overlap=None,
):
    assert sampler in ["heun", ], 'only heun sampler is supported currently'
    
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)


    sample_fn = {
        "heun": partial(sample_heun_overlap_v3, beta_d=diffusion.beta_d, beta_min=diffusion.beta_min, window_size=24),
    }[sampler]

    sampler_args = dict(
            pred_mode=diffusion.pred_mode, churn_step_ratio=churn_step_ratio, sigma_max=sigma_max
        )

    def denoiser(x_t, sigma, model_kwargs={}):
        _, denoised = diffusion.denoise(model, x_t, sigma, model_kwargs)

        return denoised
    def bridge_sample(x0, xT, t):
        dims=x0.ndim
        noise = th.randn_like(x0) 
        t = append_dims(t, dims)
        # std_t = th.sqrt(t)* th.sqrt(1 - t / self.sigma_max)
        if diffusion.pred_mode.startswith('ve'):
            std_t = t* th.sqrt(1 - t**2 / diffusion.sigma_max**2)
            mu_t= t**2 / diffusion.sigma_max**2 * xT + (1 - t**2 / diffusion.sigma_max**2) * x0
            samples = (mu_t +  std_t * noise )
        elif diffusion.pred_mode.startswith('vp'):
            logsnr_t = vp_logsnr(t, diffusion.beta_d, diffusion.beta_min)
            logsnr_T = vp_logsnr(diffusion.sigma_max, diffusion.beta_d, diffusion.beta_min)
            logs_t = vp_logs(t, diffusion.beta_d, diffusion.beta_min)
            logs_T = vp_logs(diffusion.sigma_max, diffusion.beta_d, diffusion.beta_min)

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            std_t = (-th.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
            
            samples= a_t * xT + b_t * x0 + std_t * noise
        return samples
    
    if diffusion.vae is not None:
        # breakpoint()
        with th.no_grad():
            L_A = diffusion.vae.enc_A(x_T)
            L_B_gt = diffusion.vae.enc_B(x_0)
    else:
        L_A = x_T
        L_B_gt = x_0

    # L_A = model.enc_A(x_T)
    # L_B_gt = model.enc_B(x_0)
    # breakpoint()
    L_B, path, nfe, denoised_error, x_error = sample_fn(
        denoiser,
        # bridge_sample,
        L_A,
        L_B_gt,
        sigmas,
        model_kwargs=model_kwargs,
        progress=progress,
        callback=callback,
        guidance=guidance,
        overlap=x_overlap,
        **sampler_args,
    )
    # breakpoint()
    print('nfe:', nfe)
    # x_0 = model.dec_B(L_B)
    if diffusion.vae is not None:
        # breakpoint()
        with th.no_grad():
            x_0 = diffusion.vae.dec_B(L_B)
    else:
        x_0 = L_B

    return x_0, path, nfe, denoised_error, x_error
    # return x_0.clamp(-1, 1), [x.clamp(-1, 1) for x in path], nfe


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_bridge_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, eps=1e-4, device="cpu"):
    
    sigma_t_crit = sigma_max / np.sqrt(2)
    min_start_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_t_crit ** (1 / rho)
    sigmas_second_half = (max_inv_rho + th.linspace(0, 1, n//2 ) * (min_start_inv_rho - max_inv_rho)) ** rho
    sigmas_first_half = sigma_max - ((sigma_max - sigma_t_crit)  ** (1 / rho) + th.linspace(0, 1, n - n//2 +1 ) * (eps  ** (1 / rho)  - (sigma_max - sigma_t_crit)  ** (1 / rho))) ** rho
    sigmas = th.cat([sigmas_first_half.flip(0)[:-1], sigmas_second_half])
    sigmas_bridge = sigmas**2 *(1-sigmas**2/sigma_max**2)
    return append_zero(sigmas).to(device)#, append_zero(sigmas_bridge).to(device)


def to_d(x, sigma, denoised, x_T, sigma_max,   w=1, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    grad_pxtlx0 = (denoised - x) / append_dims(sigma**2, x.ndim)
    grad_pxTlxt = (x_T - x) / (append_dims(th.ones_like(sigma)*sigma_max**2, x.ndim) - append_dims(sigma**2, x.ndim))
    gt2 = 2*sigma
    d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
    # breakpoint()
    if stochastic:
        return d, gt2
    else:
        return d


def get_d_vp(x, denoised, x_T, std_t,logsnr_t, logsnr_T, logs_t, logs_T, s_t_deriv, sigma_t, sigma_t_deriv, w, stochastic=False):
    
    a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
    b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
    
    mu_t = a_t * x_T + b_t * denoised 
    
    grad_logq = - (x - mu_t)/std_t**2 / (-th.expm1(logsnr_T - logsnr_t))
    # grad_logpxtlx0 = - (x - logs_t.exp()*denoised)/std_t**2 
    grad_logpxTlxt = -(x - th.exp(logs_t-logs_T)*x_T) /std_t**2  / th.expm1(logsnr_t - logsnr_T)

    f = s_t_deriv * (-logs_t).exp() * x
    gt2 = 2 * (logs_t).exp()**2 * sigma_t * sigma_t_deriv 
    # breakpoint()

    d = f -  gt2 * ((0.5 if not stochastic else 1)* grad_logq - w * grad_logpxTlxt)
    # d = f - (0.5 if not stochastic else 1) * gt2 * (grad_logpxtlx0 - w * grad_logpxTlxt* (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d

@th.no_grad()
def sample_heun(
    denoiser,
    x,
    gt,
    sigmas,
    pred_mode='both',
    progress=False,
    callback=None,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    churn_step_ratio=0.,
    guidance=1,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T = x
    path = [x]
    denoised_error=[]
    x_error=[]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0
    assert churn_step_ratio < 1

    if pred_mode.startswith('vp'):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
        
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

        logsnr_T = logsnr(th.as_tensor(sigma_max))
        logs_T = logs(th.as_tensor(sigma_max))
    
    for j, i in enumerate(indices):
        
        if churn_step_ratio > 0: # NOTE(xh) only at beginning refer to paper?
            # 1 step euler
            sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
            
            denoised = denoiser(x, sigmas[i] * s_in, x_T)
            if pred_mode.startswith('ve'):
                d_1, gt2 = to_d(x, sigmas[i] , denoised, x_T, sigma_max,  w=guidance, stochastic=True)
            elif pred_mode.startswith('vp'):
                d_1, gt2 = get_d_vp(x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
            
            dt = (sigma_hat - sigmas[i]) 
            x = x + d_1 * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
            
            nfe += 1
            denoised_error.append(mean_flat((denoised - gt) ** 2))
            x_error.append(mean_flat((x - gt) ** 2))
            
            path.append(x.detach().cpu())
        else:
            sigma_hat =  sigmas[i]
        
        # heun step
        denoised = denoiser(x, sigma_hat * s_in, x_T)
        # breakpoint()
        if pred_mode.startswith('ve'):
            # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
            d = to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
        elif pred_mode.startswith('vp'):
            d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
            
        nfe += 1
        denoised_error.append(mean_flat((denoised - gt) ** 2))
        x_error.append(mean_flat((x - gt) ** 2))
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            
            x = x + d * dt 
            
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, x_T)
            if pred_mode.startswith('ve'):
                # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                d_2 = to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance)
            elif pred_mode.startswith('vp'):
                d_2 = get_d_vp(x_2, denoised_2, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
            
            d_prime = (d + d_2) / 2

            # noise = th.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
            x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
            nfe += 1
            denoised_error.append(mean_flat((denoised_2 - gt) ** 2))
            x_error.append(mean_flat((x - gt) ** 2))
        # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
        # losses.append(loss)

        path.append(x.detach().cpu())
        
    return x, path, nfe, denoised_error, x_error


@th.no_grad()
def sample_heun_overlap(
    denoiser,
    bridge_sample,
    x,
    gt,
    sigmas,
    pred_mode='both',
    progress=False,
    callback=None,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    churn_step_ratio=0.,
    guidance=1,
    x_overlap=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    assert 0 in sigmas
    x_T = x
    path = [x]
    denoised_error=[]
    x_error=[]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0
    assert churn_step_ratio < 1
    if x_overlap is not None:
        overlap_len = x_overlap.shape[1]

    if pred_mode.startswith('vp'):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
        
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

        logsnr_T = logsnr(th.as_tensor(sigma_max))
        logs_T = logs(th.as_tensor(sigma_max))
    
    for j, i in enumerate(indices):
        
        if churn_step_ratio > 0: # NOTE(xh) only at beginning refer to paper?
            # 1 step euler
            sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
            
            denoised = denoiser(x, sigmas[i] * s_in, x_T)
            if pred_mode.startswith('ve'):
                d_1, gt2 = to_d(x, sigmas[i] , denoised, x_T, sigma_max,  w=guidance, stochastic=True)
            elif pred_mode.startswith('vp'):
                d_1, gt2 = get_d_vp(x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
            
            dt = (sigma_hat - sigmas[i]) 
            x = x + d_1 * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
            
            nfe += 1
            denoised_error.append(mean_flat((denoised - gt) ** 2))
            x_error.append(mean_flat((x - gt) ** 2))
            
            path.append(x.detach().cpu())
        else:
            sigma_hat =  sigmas[i]
        
        # heun step
        if x_overlap is not None:
            x[:,:overlap_len] = bridge_sample(x_overlap, x_T[:,:overlap_len], sigma_hat)
            # x[:,:overlap_len] = x_overlap
        denoised = denoiser(x, sigma_hat * s_in, x_T)
        # breakpoint()
        if pred_mode.startswith('ve'):
            # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
            d = to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
        elif pred_mode.startswith('vp'):
            d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
            
        nfe += 1
        denoised_error.append(mean_flat((denoised - gt) ** 2))
        x_error.append(mean_flat((x - gt) ** 2))
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            
            x = x + d * dt 
            
        else:
            # Heun's method
            x_2 = x + d * dt
            if x_overlap is not None:
                x_2[:,:overlap_len] = bridge_sample(x_overlap, x_T[:,:overlap_len], sigmas[i + 1])
                # x_2[:,:overlap_len] = x_overlap
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, x_T)
            if pred_mode.startswith('ve'):
                # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                d_2 = to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance)
            elif pred_mode.startswith('vp'):
                d_2 = get_d_vp(x_2, denoised_2, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
            
            d_prime = (d + d_2) / 2

            # noise = th.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
            x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
            nfe += 1
            denoised_error.append(mean_flat((denoised_2 - gt) ** 2))
            x_error.append(mean_flat((x - gt) ** 2))
        # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
        # losses.append(loss)

        path.append(x.detach().cpu())
        
    if x_overlap is not None:
        # x[:,overlap_len:] = bridge_sample(x_overlap, x_T[:,:overlap_len], sigma_hat)
        x[:,:overlap_len] = x_overlap
    return x, path, nfe, denoised_error, x_error

@th.no_grad()
def sample_heun_overlap_v2(
    denoiser,
    # bridge_sample,
    x,
    gt,
    sigmas,
    overlap,
    window_size,
    pred_mode='both',
    progress=False,
    callback=None,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    churn_step_ratio=0.,
    guidance=1,
):
    
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    def get_windows(x_input):
        assert x_input.shape[0] == 1
        windows = [x_input[0][s:s+window_size] for (s,e) in overlap]
        return th.stack(windows, dim=0)
    
    def cat_overlap(x_output):
        assert x_output.shape[0] == len(overlap)
        x_cat = []
        interval_last=0
        for i, (s,e) in enumerate(overlap[1:]):
            interval = e-s
            x_cat.append(x_output[i,interval_last:window_size-interval])
            x_cat.append((x_output[i,window_size-interval:] + x_output[i+1,:interval])/2.)
            interval_last = interval
        x_cat.append(x_output[-1,interval_last:])
        x_cat = th.cat(x_cat, dim=0)
        return x_cat.unsqueeze(0)
    assert 0 in sigmas
    x = get_windows(x)
    x_T = x
    path = [x]
    denoised_error=[]
    x_error=[]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0
    assert churn_step_ratio < 1
    breakpoint()

    if pred_mode.startswith('vp'):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
        
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

        logsnr_T = logsnr(th.as_tensor(sigma_max))
        logs_T = logs(th.as_tensor(sigma_max))
    
    for j, i in enumerate(indices):
        
        if churn_step_ratio > 0: # NOTE(xh) only at beginning refer to paper?
            # 1 step euler
            sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
            
            denoised = denoiser(x, sigmas[i] * s_in, x_T)
            if pred_mode.startswith('ve'):
                d_1, gt2 = to_d(x, sigmas[i] , denoised, x_T, sigma_max,  w=guidance, stochastic=True)
            elif pred_mode.startswith('vp'):
                d_1, gt2 = get_d_vp(x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
            
            dt = (sigma_hat - sigmas[i]) 
            x = x + d_1 * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
            
            nfe += 1
            denoised_error.append(mean_flat((denoised - gt) ** 2))
            x_error.append(mean_flat((x - gt) ** 2))
            
            path.append(x.detach().cpu())
        else:
            sigma_hat =  sigmas[i]
        
        # heun step
        denoised = denoiser(x, sigma_hat * s_in, x_T)
        denoised = cat_overlap(denoised)
        denoised_error.append(mean_flat((denoised - gt) ** 2))
        denoised = get_windows(denoised)
        breakpoint()
        if pred_mode.startswith('ve'):
            # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
            d = to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
        elif pred_mode.startswith('vp'):
            d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
            
        nfe += 1
        # breakpoint()
        x_error.append(mean_flat((cat_overlap(x) - gt) ** 2))
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            
            x = x + d * dt 
            
        else:
            # Heun's method
            x_2 = x + d * dt
            
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, x_T)
            denoised_2 = cat_overlap(denoised_2)
            denoised_error.append(mean_flat((denoised_2 - gt) ** 2))
            denoised_2 = get_windows(denoised_2)
            if pred_mode.startswith('ve'):
                # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                d_2 = to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance)
            elif pred_mode.startswith('vp'):
                d_2 = get_d_vp(x_2, denoised_2, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
            
            d_prime = (d + d_2) / 2

            # noise = th.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
            x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
            nfe += 1
            x_error.append(mean_flat((cat_overlap(x) - gt) ** 2))
        # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
        # losses.append(loss)

        path.append(x.detach().cpu())
        
    x = cat_overlap(x)
    return x, path, nfe, denoised_error, x_error


@th.no_grad()
def sample_heun_overlap_v3(
    denoiser,
    # bridge_sample,
    x,
    gt,
    sigmas,
    model_kwargs,
    overlap,
    window_size,
    pred_mode='both',
    progress=False,
    callback=None,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    churn_step_ratio=0.,
    guidance=1,
):
    
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    def get_windows(x_input):
        assert x_input.shape[0] == 1
        windows = [x_input[0][s:s+window_size] for (s,e) in overlap]
        return th.stack(windows, dim=0)
    
    def cat_overlap(x_output):
        assert x_output.shape[0] == len(overlap)
        x_cat = []
        interval_last=0
        for i, (s,e) in enumerate(overlap[1:]):
            interval = e-s
            x_cat.append(x_output[i,interval_last:window_size-interval])
            x_cat.append((x_output[i,window_size-interval:] + x_output[i+1,:interval])/2.)
            interval_last = interval
        x_cat.append(x_output[-1,interval_last:])
        x_cat = th.cat(x_cat, dim=0)
        return x_cat.unsqueeze(0)
    assert 0 in sigmas
    cond = get_windows(model_kwargs["cond"])
    x = get_windows(x)
    x_T = x
    path = [x]
    denoised_error=[]
    x_error=[]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0
    assert churn_step_ratio < 1
    # breakpoint()

    if pred_mode.startswith('vp'):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
        
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

        logsnr_T = logsnr(th.as_tensor(sigma_max))
        logs_T = logs(th.as_tensor(sigma_max))
    
    for j, i in enumerate(indices):
        
        if churn_step_ratio > 0: # NOTE(xh) only at beginning refer to paper?
            # 1 step euler
            sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
            
            denoised = denoiser(x, sigmas[i] * s_in, x_T)
            if pred_mode.startswith('ve'):
                d_1, gt2 = to_d(x, sigmas[i] , denoised, x_T, sigma_max,  w=guidance, stochastic=True)
            elif pred_mode.startswith('vp'):
                d_1, gt2 = get_d_vp(x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
            
            dt = (sigma_hat - sigmas[i]) 
            x = x + d_1 * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
            
            nfe += 1
            denoised_error.append(mean_flat((denoised - gt) ** 2))
            x_error.append(mean_flat((x - gt) ** 2))
            
            path.append(x.detach().cpu())
        else:
            sigma_hat =  sigmas[i]
        
        # heun step
        denoised = denoiser(x, sigma_hat * s_in, {'cond':cond})
        denoised = cat_overlap(denoised)
        denoised_error.append(mean_flat((denoised - gt) ** 2))
        denoised = get_windows(denoised)
        # breakpoint()
        if pred_mode.startswith('ve'):
            # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
            d = to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
        elif pred_mode.startswith('vp'):
            d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
            
        nfe += 1
        # breakpoint()
        x_error.append(mean_flat((cat_overlap(x) - gt) ** 2))
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            
            x = x + d * dt 
            
        else:
            # Heun's method
            x_2 = x + d * dt
            
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, {'cond':cond})
            denoised_2 = cat_overlap(denoised_2)
            denoised_error.append(mean_flat((denoised_2 - gt) ** 2))
            denoised_2 = get_windows(denoised_2)
            if pred_mode.startswith('ve'):
                # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                d_2 = to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance)
            elif pred_mode.startswith('vp'):
                d_2 = get_d_vp(x_2, denoised_2, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
            
            d_prime = (d + d_2) / 2

            # noise = th.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
            x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
            nfe += 1
            x_error.append(mean_flat((cat_overlap(x) - gt) ** 2))
        # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
        # losses.append(loss)

        path.append(x.detach().cpu())
        
    x = cat_overlap(x)
    return x, path, nfe, denoised_error, x_error



@th.no_grad()
def forward_sample(
    x0,
    y0,
    sigma_max,
    ):

    ts = th.linspace(0, sigma_max, 120)
    x = x0
    # for t, t_next in zip(ts[:-1], ts[1:]):
    #     grad_pxTlxt = (y0 - x) / (append_dims(th.ones_like(ts)*sigma_max**2, x.ndim) - append_dims(t**2, x.ndim))
    #     dt = (t_next - t) 
    #     gt2 = 2*t
    #     x = x + grad_pxTlxt * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
    path = [x]
    for t in ts:
        std_t = th.sqrt(t)* th.sqrt(1 - t / sigma_max)
        mu_t= t / sigma_max * y0 + (1 - t / sigma_max) * x0
        xt = (mu_t +  std_t * th.randn_like(x0) )
        path.append(xt)

    path.append(y0)

    return path


