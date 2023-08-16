"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
from PIL import Image
import torchvision
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

import cv2
from torchvision.transforms import ToPILImage


to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

from .clip.base_clip import CLIPEncoder

from .arcface.model import IDLoss

from einops import rearrange
from PIL import Image
import os

import dlib
from skimage import transform as trans
from skimage import io


def get_points_and_rec(img, detector, shape_predictor, size_threshold=999):
    dets = detector(img, 1)
    if len(dets) == 0:
        return None, None
    
    all_points = []
    rec_list = []
    for det in dets:
        if isinstance(detector, dlib.cnn_face_detection_model_v1):
            rec = det.rect # for cnn detector
        else:
            rec = det
        if rec.width() > size_threshold or rec.height() > size_threshold: 
            break
        rec_list.append(rec)
        shape = shape_predictor(img, rec) 
        single_points = []
        for i in range(5):
            single_points.append([shape.part(i).x, shape.part(i).y])
        all_points.append(np.array(single_points))
    if len(all_points) <= 0:
        return None, None
    else:
        return all_points, rec_list


def align_and_save(img, src_points, template_path, template_scale=1, img_size=256):
    out_size = (img_size, img_size)
    reference = np.load(template_path) / template_scale * (img_size / 256)

    for idx, spoint in enumerate(src_points):
        tform = trans.SimilarityTransform()
        tform.estimate(spoint, reference)
        M = tform.params[0:2,:]
        return M, img.shape


def align_and_save_dir(src_path, template_path='./pretrain_models/FFHQ_template.npy', template_scale=4, use_cnn_detector=True, img_size=256):
    if use_cnn_detector:
        detector = dlib.cnn_face_detection_model_v1('./pretrain_models/mmod_human_face_detector.dat')
    else:
        detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('./pretrain_models/shape_predictor_5_face_landmarks.dat')

    img_path = src_path
    img = dlib.load_rgb_image(img_path)

    points, rec_list = get_points_and_rec(img, detector, sp)
    if points is not None:
        return align_and_save(img, points, template_path, template_scale, img_size=img_size)


def get_tensor_M(src_path):
    M, s = align_and_save_dir(src_path)
    h, w = s[0], s[1]
    a = torch.Tensor(
        [
            [2/(w-1), 0, -1],
            [0, 2/(h-1), -1],
            [0, 0, 1]
        ]
    )
    Mt = torch.Tensor(
        [  
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    )
    Mt[:2, :] = torch.Tensor(M)
    Mt = torch.inverse(Mt)
    h, w = 256, 256
    b = torch.Tensor(
        [
            [2/(w-1), 0, -1],
            [0, 2/(h-1), -1],
            [0, 0, 1]
        ]
    )
    b = torch.inverse(b)
    Mt = a.matmul(Mt)
    Mt = Mt.matmul(b)[:2].unsqueeze(0)
    return Mt


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", add_condition_mode="face_id", ref_path=None, add_ref_path=None, no_freedom=False, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.add_condition_mode = add_condition_mode
        self.no_freedom = no_freedom
        if self.add_condition_mode == "face_id":
            self.idloss = IDLoss(ref_path=add_ref_path).cuda()
            M = get_tensor_M(ref_path)
            self.grid = F.affine_grid(M, (1, 3, 256, 256), align_corners=True).cuda()
        elif self.add_condition_mode == "style":
            image_encoder = CLIPEncoder(need_ref=True, ref_path=add_ref_path).cuda()
            self.image_encoder = image_encoder.requires_grad_(True)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples, intermediates

    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        
        device = self.model.betas.device
        b = shape[0]

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        time_range_reverse = time_range[::-1]
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if self.add_condition_mode == "style":
                outs = self.p_sample_ddim_style(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          dynamic_threshold=dynamic_threshold)
            elif self.add_condition_mode == "face_id":
                outs = self.p_sample_ddim_pose(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def p_sample_ddim_style(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        if 70 > index >= 40:
            repeat = 3
        else:
            repeat = 1 

        start = 70
        end = 30
        if self.no_freedom:
            start = end = -10

        for j in range(repeat):

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0)
                residual = self.image_encoder.get_gram_matrix_residual(D_x0_t)
                norm = torch.linalg.norm(residual)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
                rho = (correction * correction).mean().sqrt().item() * unconditional_guidance_scale 
                rho = rho / (norm_grad * norm_grad).mean().sqrt().item() * 0.2

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

            if start > index >= end:
                x_prev = x_prev - rho * norm_grad.detach()
            
            x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

        return x_prev.detach(), pred_x0.detach()

    def p_sample_ddim_pose(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)
        # self.model.decode_first_stage.decoder.requires_grad_(True)

        repeat = 1 
        start = 40
        end = -10
        if self.no_freedom:
            start = end = -10

        for j in range(repeat):

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            if start > index >= end:
                    D_x0_t = self.model.decode_first_stage(pred_x0)
                    warp_D_x0_t = F.grid_sample(D_x0_t, self.grid, align_corners=True)
                    residual = self.idloss.get_residual(warp_D_x0_t)
                    norm = torch.linalg.norm(residual)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
                    rho = (correction * correction).mean().sqrt().item() * unconditional_guidance_scale 
                    rho = rho / (norm_grad * norm_grad).mean().sqrt().item() * 0.08  # 0.08

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

            if start > index >= end:
                x_prev = x_prev - rho * norm_grad.detach()
            
            x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

        return x_prev.detach(), pred_x0.detach()

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec