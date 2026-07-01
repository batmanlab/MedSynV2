import argparse
import math
import copy
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import AdamW
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange, repeat
from torchdiffeq import odeint
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from dataloader import  cache_transformed_train_data_dit_112_patho
import glob, os
from einops_exts import check_shape

from accelerate import Accelerator
from video_transformer_v4_gated import VDiT_models

torch.backends.cudnn.benchmark = True

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(np.pi / 2 * t) + 1))

def get_alpha_cum(t):
    return torch.where(t >= 0, torch.cos((t + 0.008) / 1.008 * math.pi / 2)**2, 1.0).clamp(1e-4, 1-1e-4)

def get_z_t(x_0, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*eps
    return x_t, eps

def get_eps_x_t(x_0, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    eps = (x_t - torch.sqrt(alpha_cum)*x_0)/torch.sqrt(1-alpha_cum)
    return eps

def get_x0_x_t(eps, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    x_0 = (x_t - eps * torch.sqrt(1-alpha_cum)) / torch.sqrt(alpha_cum)
    return x_0

def get_x0_v(v, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    sigma = 1 - alpha_cum
    return x_t - sigma*v

def get_v_x0(x0, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    sigma = 1 - alpha_cum
    v = (x_t - x0)/sigma
    return v

def get_z_t_(x_0, t):
    alpha_cum = get_alpha_cum(t)[:,None]
    return torch.sqrt(alpha_cum)*x_0, torch.sqrt(1-alpha_cum)

def get_z_t_via_z_tp1(x_0, z_tp1, t, t_p1):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    alpha_cum_p1 = get_alpha_cum(t_p1)[:, None, None, None, None]
    beta_p1 = 1 - alpha_cum_p1/alpha_cum
    mean_0 = torch.sqrt(alpha_cum)*beta_p1/(1-alpha_cum_p1)
    mean_tp1 = torch.sqrt(1-beta_p1)*(1-alpha_cum)/(1-alpha_cum_p1)

    var = (1-alpha_cum)/(1-alpha_cum_p1)*beta_p1

    return mean_0*x_0 + mean_tp1*z_tp1, var

def ddim_sample(x_0, z_tp1, t, t_p1):
    epsilon = get_eps_x_t(x_0, z_tp1, t_p1)
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    x_t = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*epsilon
    return x_t

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


# helpers functions

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            num_frames,
            text_use_bert_cls=False,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            use_dynamic_thres=False,  # from the Imagen paper
            dynamic_thres_percentile=0.9,
            volume_depth=128,
            ddim_timesteps=50,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.volume_depth = volume_depth
        self.num_timesteps = timesteps
        self.loss_type = loss_type

        self.ddim_timesteps = ddim_timesteps
        betas = cosine_beta_schedule(timesteps)
        
        # text conditioning parameters
        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)

    def p_mean_variance(self, x, t, clip_denoised: bool, indexes=None, cond=None, cond_scale=1.):
        # x_recon is now correctly predicting x0 directly (z_theta)
        x_recon = self.denoise_fn.forward(x, t, text=cond)
    
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance = get_z_t_via_z_tp1(
            x_recon, x, 
            (t - 1) * 1.0 / (self.num_timesteps - 1.0),
            (t * 1.0) / (self.num_timesteps - 1.0)
        )
        return model_mean, posterior_variance

    @torch.inference_mode()
    def p_sample(self, x, t, indexes=None, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(
            x=x, t=t, indexes=indexes, clip_denoised=clip_denoised,
            cond=cond, cond_scale=cond_scale
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, 1, self.num_frames, 1, 1)
        return model_mean + nonzero_mask * (model_variance**0.5) * noise

    @torch.inference_mode()
    def p_sample_ddim(self, x, t, t_minus, indexes=None, cond=None, cond_scale=1., clip_denoised=False):
        b, *_, device = *x.shape, x.device
        
        x_recon = self.denoise_fn.forward(x, t, text=cond, use_eval=True)
        if cond_scale != 1:
            x_recon, x_recon_null = x_recon
            eps = get_eps_x_t(x_recon, x, t)
            eps_null = get_eps_x_t(x_recon_null, x, t)
            final_eps = eps_null + (eps - eps_null) * cond_scale
            x_recon = get_x0_x_t(final_eps, x, t)
            
        if t[0] < int(self.num_timesteps / self.ddim_timesteps):
            x = x_recon
        else:
            t_minus = torch.clip(t_minus, min=0.0)
            x = ddim_sample(x_recon, x, (t_minus * 1.0) / (self.num_timesteps), (t * 1.0) / (self.num_timesteps))
        return x

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1., use_ddim=True):
        device = self.betas.device
        bsz = shape[0]

        if use_ddim:
            time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_timesteps))
        else:
            time_steps = range(0, self.num_timesteps)

        img = torch.randn(shape, device=device)
        indexes = []
        for b in range(bsz):
            index = np.arange(self.num_frames)
            indexes.append(torch.from_numpy(index))
        indexes = torch.stack(indexes, dim=0).long().to(device)
        
        for i, t in enumerate(tqdm(reversed(time_steps), desc='sampling loop time step', total=len(time_steps))):
            time = torch.full((bsz,), t, device=device, dtype=torch.float16)

            if use_ddim:
                time_minus = time - int(self.num_timesteps / self.ddim_timesteps)
                img = self.p_sample_ddim(img, time, time_minus, indexes=indexes, cond=cond, cond_scale=cond_scale)
            else:
                img = self.p_sample(img, time, indexes=indexes, cond=cond, cond_scale=cond_scale)
        return unnormalize_img(img)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def p_losses(self, x_start, t, indexes=None, cond=None, noise=None, lpips=None, **kwargs):
        # Apply cosmap to times mapping
        t_mapped = cosmap(t) if 'cosmap' in globals() else t
        b, c, f, h, w, device = *x_start.shape, x_start.device
        
        # z_0 is the pure target (data), x_cond is the condition stack
        z_0, x_cond = x_start[:, :4], x_start[:, 4:]
        
        noise = torch.randn_like(z_0)
        
        # Safely reshape mapped t to correctly broadcast with 5D latent dimensions
        t_view = t_mapped.view(b, 1, 1, 1, 1)
        
        # Calculate z_t 
        noised = t_view * z_0 + (1. - t_view) * noise
        
        # Network directly predicts z_theta (x-prediction)
        z_theta = self.denoise_fn(torch.cat([noised, x_cond], dim=1), t_mapped * self.num_timesteps, text=cond)
        
        # Formulate velocities according to user math block
        # eps = 1e-7  # Keep stability avoiding divide-by-zero
        # v_theta  = (z_theta - noised) / (0.0 - t_view - eps)
        # v_target = (z_0 - noised)     / (0.0 - t_view - eps)

        # the above commented lines are mathematically equivalent to the following simplified version:
        # v_theta = z_theta
        # v_target = z_0
        
        loss = F.mse_loss(z_theta, z_0)
        
        if lpips is not None:
            n_frames = min(50, f)
            frame_indices = torch.randperm(f)[:n_frames]
            lpips_eff = 0.1
            
            # Applying LPIPS loss component on constructed velocities
            for j in range(c):
                flow_      = z_0[:, j:j+1].repeat(1, 3, 1, 1, 1)
                pred_flow_ = z_theta[:, j:j+1].repeat(1, 3, 1, 1, 1)
                
                flow_      = flow_[:, :, frame_indices, :, :].reshape(-1, 3, w, h)
                pred_flow_ = pred_flow_[:, :, frame_indices, :, :].reshape(-1, 3, w, h)
                
                p_loss = lpips(pred_flow_.contiguous(), flow_.contiguous().detach())
                loss = loss + lpips_eff * p_loss.mean()

        return loss
        
    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, model, x_t, x_cond=None, *, times, eps=1e-7, cond=None, cond_scale=1.):
        # Updated signature to accept unconcatenated latents & conditions
        batch = x_t.shape[0]
        times = rearrange(times, '... -> (...)')
        if times.numel() == 1:
            times = repeat(times, '1 -> b', b=batch)
            
        model_in = torch.cat([x_t, x_cond], dim=1) if x_cond is not None else x_t
        
        # Predict z_theta
        z_theta = self.denoise_fn(model_in, times * self.num_timesteps, text=cond)
        
        # Reformulate flow mathematically from z_theta prediction map 
        t_view = times.view(-1, *((1,) * (x_t.ndim - 1)))
        flow = (z_theta - x_t) / (0.0 - t_view - eps)
        
        return flow

    @torch.inference_mode()
    @torch.autocast("cuda")
    def sample(self, x_cond=None, cond=None, cond_scale=1., batch_size=16, DDIM=True):
        batch_size = cond.shape[0] if exists(cond) else batch_size
        device = x_cond.device
        image_size = self.image_size
        channels = self.channels - 4
        num_frames = self.num_frames

        def ode_fn(t, x):
            t_batch = torch.full(
                (x.shape[0],),
                t,
                device=device
            )

            t_cos = cosmap(t_batch) if 'cosmap' in globals() else t_batch
            
            # Calculate continuous flow
            flow = self.predict_flow(
                self.denoise_fn,
                x_t=x,          # 4-Channel state
                x_cond=x_cond,  # 4-Channel conditioning 
                times=t_cos,
                cond=cond,
                cond_scale=cond_scale
            )
            return flow

        times = torch.linspace(
            0.0,
            1.0,
            self.ddim_timesteps,
            device=device
        )

        # ----------------------------------------
        # Initial noise
        # ----------------------------------------
        x0 = torch.randn(
            batch_size,
            channels,
            num_frames,
            image_size,
            image_size,
            device=device
        )

        trajectory = odeint(
            ode_fn,
            x0,
            times,
            atol=1e-5,
            rtol=1e-5,
            method="midpoint" if DDIM else "dopri5"
        )
        sampled_data = trajectory[-1]
        return sampled_data

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.rand((b), device=device).float()
        return self.p_losses(x, t, *args, **kwargs)


# trainer class

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images


# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(x_recon):
    x_recon = x_recon.clamp(-1, 1)
    return (x_recon + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


# trainer class


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            num_frames=16,
            train_batch_size=32,
            train_lr=1e-4,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            amp=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            num_sample_rows=4,
            max_grad_norm=None
    ):
        super().__init__()

        self.ema_model = diffusion_model
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        self.num_frames = diffusion_model.num_frames

        train_files = []
        folder_latent_1 = os.path.join(args.mask_folder, "lobe_latents_112_")
        folder_latent_2 = os.path.join(args.mask_folder, "airway_latents_112_")
        folder_latent_3 = os.path.join(args.mask_folder, "vessel_latents_112_")
        folder_latent_4 = os.path.join(args.mask_folder, "cons_latents_112_")
        folder_latent_5 = os.path.join(args.mask_folder, "ggo_latents_112_")
        folder_latent_6 = os.path.join(args.mask_folder, "pleffu_latents_112_")
        folder_latent_7 = os.path.join(args.mask_folder, "perieffu_latents_112_")
        folder_latent_8 = os.path.join(args.mask_folder, "heart_latents_112_")
        folder_latent_9 = os.path.join(args.mask_folder, "nodule_latents_112_")
        items = list(sorted([item.replace('.npy', '') for item in os.listdir(folder)[:]]))
        
        for img_dir in items[:]:
            if os.path.exists(os.path.join(folder, img_dir+'.npy')):
                text_real_path = os.path.join(folder, img_dir+'.npy')
            else:
                continue
            latent_name = img_dir 
            sample = {
                "text_real": text_real_path,
            }

            # EXACT filename mapping (do NOT change semantics)
            feat_paths = {
                "lobe_feat": os.path.join(
                    folder_latent_1,
                    latent_name + "_processed_Reg_lobes.npy"
                ),
                "airway_feat": os.path.join(
                    folder_latent_2,
                    latent_name + "_processed_Reg_airways.npy"
                ),
                "vessel_feat": os.path.join(
                    folder_latent_3,
                    latent_name + "_processed_Reg_lung_vessels.npy"
                ),
                "cons_feat": os.path.join(
                    folder_latent_4,
                    latent_name + "_processed_Reg_consolidation.npy"
                ),
                "ggo_feat": os.path.join(
                    folder_latent_5,
                    latent_name + "_processed_Reg_ggo.npy"
                ),
                # IMPORTANT: same file for two semantic features
                "pleffu_feat": os.path.join(
                    folder_latent_6,
                    latent_name + "_processed_Reg_pleffu.npy"
                ),
                "perieffu_feat": os.path.join(
                    folder_latent_7,
                    latent_name + "_processed_Reg_perieffu.npy"
                ),
                "heart_feat": os.path.join(
                    folder_latent_8,
                    latent_name + "_processed_Reg_heart.npy"
                ),
                "nodule_feat": os.path.join(
                    folder_latent_9,
                    latent_name + "_processed_Reg_nodule.npy"
                ),
            }

            # Only insert keys if files exist 
            for k, p in feat_paths.items():
                if os.path.exists(p):
                    sample[k] = p

            train_files.append(sample)
        self.ds = cache_transformed_train_data_dit_112_patho(train_files=train_files, shape=[image_size, image_size, image_size], crop_shape=[image_size, 64, 64])

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = data.DataLoader(self.ds, batch_size=train_batch_size, num_workers=1, shuffle=False, pin_memory=True)
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), weight_decay=0.01)

        self.step = 0

        self.amp = amp
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        if amp:
            mixed_precision = "fp16"
        else:
            mixed_precision = "fp32"

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            mixed_precision=mixed_precision,
        )

        self.ema_model = self.accelerator.prepare(self.ema_model)


        prompt_lobe = torch.from_numpy(np.load('../extra_prompts_gen/ana_lobe.npy'))
        prompt_airway = torch.from_numpy(np.load('../extra_prompts_gen/ana_airway.npy'))
        prompt_vessel = torch.from_numpy(np.load('../extra_prompts_gen/ana_vessel.npy'))

        prompt_cons = torch.from_numpy(np.load('../extra_prompts_gen/path_cons.npy'))
        prompt_ggo = torch.from_numpy(np.load('../extra_prompts_gen/path_ggo.npy'))
        prompt_perieffu = torch.from_numpy(np.load('../extra_prompts_gen/path_perieffu.npy'))
        prompt_pleffu = torch.from_numpy(np.load('../extra_prompts_gen/path_pleffu.npy'))
        prompt_heart = torch.from_numpy(np.load('../extra_prompts_gen/ana_heart.npy'))
        prompt_nodule = torch.from_numpy(np.load('../extra_prompts_gen/path_nodule.npy'))
        
        prompt_nothing = torch.from_numpy(np.load('../extra_prompts_gen/empty_msk.npy'))

        self.prompts = [prompt_lobe, prompt_airway, prompt_vessel, prompt_heart, prompt_cons, prompt_ggo, prompt_perieffu, prompt_pleffu, prompt_nodule, prompt_nothing]
        
        self.image_means = torch.tensor([0.8824114, -0.051992204, 1.1762542, -7.0781837], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1).to(self.accelerator.device)
        self.image_stds  = torch.tensor([3.4852407, 5.1318316, 4.399875, 3.8425775], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1).to(self.accelerator.device)
        
        lobe_means = torch.tensor([1.6398076, 3.988608, 1.0285985, -10.314252], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        lobe_stds  = torch.tensor([1.9901804, 2.4358141, 1.884654, 2.492379], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        
        airway_means = torch.tensor([2.185043, 5.434302, 1.486879, -14.069491], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        airway_stds  = torch.tensor([1.0382589, 1.1419985, 0.6198819, 1.2998749], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)

        vessel_means = torch.tensor([1.8914073, 4.157799, 1.571526, -11.400889], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        vessel_stds  = torch.tensor([1.6818109, 1.7393458, 1.1529821,  2.427308], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)

        heart_means = torch.tensor([1.677869, 5.143252, 0.9597176, -12.662287], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        heart_stds  = torch.tensor([2.3454425, 1.4271269, 2.0748055, 2.4220674], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)

        cons_means = torch.tensor([-0.8596495, 9.748561, -4.030914, 0.20274788], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        cons_stds  = torch.tensor([0.13218798, 1.1483095, 0.46476543, 0.27000445], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)

        ggo_means = torch.tensor([2.120977, 5.287014, 1.4342078, -13.763023], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        ggo_stds  = torch.tensor([1.1449654, 1.3086046, 0.838108, 1.6218491], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)

        
        perieffu_means = torch.tensor([2.2425787, 5.654407, 1.4436094, -14.502524], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        perieffu_stds  = torch.tensor([0.8545741, 0.98988974, 0.44174922, 1.1419139], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
    
        pleffu_means = torch.tensor([2.1982493, 5.633169, 1.4147488, -14.4075], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        pleffu_stds  = torch.tensor([0.96593255, 1.0451757, 0.67296726, 1.2834657], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)

        nodule_means = torch.tensor([1.8867295, 5.4893284, 1.1319814, -13.561167], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        nodule_stds  = torch.tensor([2.182885, 1.1296502, 1.6234175, 2.117792], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        
        nothing_means = torch.tensor([0, 0, 0, 0], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)
        nothing_stds  = torch.tensor([1, 1, 1, 1], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1)#.to(self.accelerator.device)

        self.means = [lobe_means, airway_means, vessel_means, heart_means, cons_means, ggo_means, perieffu_means, pleffu_means, nodule_means, nothing_means]
        self.stds = [lobe_stds,   airway_stds,  vessel_stds,  heart_stds,  cons_stds,  ggo_stds,  perieffu_stds,  pleffu_stds,  nodule_stds,  nothing_stds]

    def load(self, milestone, **kwargs):
        if milestone == -1:
            path = 'final_ckpt' 
        self.step = 1 
        self.accelerator.load_state(os.path.join(self.results_folder, path), strict=True)


    def train(
            self,
            prob_focus_present=0.,
            focus_present_mask=None,
            log_fn=noop
    ):
        assert callable(log_fn)

        self.results_folder = save_path
        if not os.path.exists(self.results_folder) and self.accelerator.is_main_process:
            os.mkdir(self.results_folder)
        self.ema_model.eval()
        self.ema_model.half()
        feat_keys = [
            "lobe_feat",
            "airway_feat",
            "vessel_feat",
            "heart_feat",
            "cons_feat",
            "ggo_feat",
            "perieffu_feat",
            "pleffu_feat",
            "nodule_feat",
        ]
        for i, data in enumerate(self.dl):

            feats = [data.get(k) for k in feat_keys]
            for idx_feat in range(len(feat_keys)):
                with torch.no_grad():
                    try:
                        chosen_feat = feats[idx_feat].squeeze(dim=1) 
                    except Exception as  e:
                        print(e)
                        continue
                    means = self.means[idx_feat]
                    stds = self.stds[idx_feat]
                    chosen_feat = ((chosen_feat - means) / stds).to(self.accelerator.device).half()
                    chosen_text = self.prompts[idx_feat].to(self.accelerator.device)

                    if include_report:
                        text = data['text_real']
                        text = text.to(self.accelerator.device)
                        text = torch.cat([chosen_text, text, ], dim=1)
                    else:
                        text = chosen_text

                    with torch.no_grad():
                        file_name = data['text_real_meta_dict']['filename_or_obj'][0].split('/')[-1]
                        video_path = os.path.join(self.results_folder, str(f'{file_name}_{idx_feat}.npy'))
                        if os.path.exists(video_path):
                            continue

                        num_samples = self.num_sample_rows ** 2
                        batches = num_to_groups(num_samples, self.batch_size)
                        if hasattr(self.ema_model, 'module'):
                            all_videos_list = list(
                                map(lambda n: self.ema_model.module.sample(x_cond=chosen_feat, batch_size=n, cond=text, cond_scale=1), batches))
                        else:
                            all_videos_list = list(
                                map(lambda n: self.ema_model.sample(x_cond=chosen_feat, batch_size=n, cond=text, cond_scale=1), batches))
                        all_videos_list = torch.cat(all_videos_list, dim=0)
                        np.save(os.path.join(self.results_folder, str(f'{file_name}_{idx_feat}')),
                                all_videos_list.cpu().numpy())


def main(args):
    model = VDiT_models['VDiT_XL_122_24'](input_size=(112, 56, 56), dtype=torch.float16, caption_channels=768, in_channels=8, out_channels=4)

    diffusion_model = GaussianDiffusion(
            denoise_fn=model,
            image_size=56,
            num_frames=112,
            text_use_bert_cls=False,
            channels=8,
            timesteps=1000,
            loss_type='x0',
            use_dynamic_thres=False,  
            dynamic_thres_percentile=0.995,
            volume_depth=112,
            ddim_timesteps=50,
        )
        

    trainer = Trainer(diffusion_model=diffusion_model,
                    folder=args.text_feature_folder,
                    mask_folder=args.mask_folder,
                    ema_decay=0.9999,
                    num_frames=112,
                    train_batch_size=1,
                    train_lr=1e-4,
                    train_num_steps=1000000,
                    gradient_accumulate_every=8,
                    amp=True,
                    step_start_ema=10000,
                    update_ema_every=1,
                    save_and_sample_every=500,
                    results_folder=args.pretrain_model_path,
                    num_sample_rows=1,
                    max_grad_norm=1.0)

    trainer.load(-1)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_feature_folder', type=str, default='./text_feature')
    parser.add_argument('--mask_folder', type=str, default='./mask_feature')
    parser.add_argument('--pretrain_model_path', type=str, default='./model/medsynv2_dit_x0.pth')
    parser.add_argument('--save_path', type=str, default='./tmp/medsynv2_dit_x0_results_reportonly')
    parser.add_argument('--include_report', action='store_true', default=False, help='whether to include the report text in the conditioning')
    args = parser.parse_args()
    save_path = args.save_path
    include_report = args.include_report

    os.makedirs(save_path, exist_ok=True)
    main(args)
