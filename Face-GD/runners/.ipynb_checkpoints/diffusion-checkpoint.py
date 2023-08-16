import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.denoising import efficient_generalized_steps, simple_diffusion, complex_diffusion, plus_diffusion, grad_diffusion, ddpm_diffusion, grad_ddpm_diffusion, clip_ddpm_diffusion, gp_ddpm_diffusion, gp_ddim_diffusion, ddgm_linear_diffusion, clip_ddim_diffusion, ilvr_diffusion, ccdf_diffusion, parse_ddim_diffusion, sketch_ddim_diffusion, landmark_ddim_diffusion, arcface_ddim_diffusion, arcface_landmark_ddim_diffusion, style_ddim_diffusion, style_transfer_ddim_diffusion, clip_parser_ddim_diffusion, clip_parser_edit_ddim_diffusion, clip_edit_ddim_diffusion, parse_clip_ddim_diffusion

import torchvision.utils as tvu

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth


def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def tensor2uint(img):
    img = (img + 1.0) / 2.0
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level


def get_gray_gaussian_noisy_img(img, noise_level):
    b, c, h, w = img.size()
    return img + torch.randn((b, 1, h, w)).cuda() * noise_level


def get_3d_gaussian_noisy_img(img, noise_level):
    b, c, h, w = img.size()
    L = noise_level
    D = np.diag(np.random.rand(3))
    U = orth(np.random.rand(3, 3))
    conv = np.dot(np.dot(np.transpose(U), D), U)
    noise = np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), (h, w)).astype(np.float32)
    noise = single2tensor4(noise)
    return img + noise.cuda()


def get_poisson_noisy_img(img):
    img = (img + 1.0) / 2.0
    img = img.cpu().numpy()

    img = np.clip((img * 255.0).round(), 0, 255) / 255.
#     vals = 10 ** (2 * random.random() + 2.0)  # [2, 4]
    vals = 10 ** 2
    img = np.random.poisson(img * vals).astype(np.float32) / vals
    img = np.clip(img, 0.0, 1.0)

    img = img * 2.0 - 1.0

    return torch.from_numpy(img).float().cuda()


def get_speckel_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level * img


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':    
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                #ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
                
            
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size, ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config

        #get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)
        
        device_count = torch.cuda.device_count()
        
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        

        ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by, torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        elif deg[:3] == 'ccs':
            cs_ratio = int(deg[3:]) / 100.0
            from functions.svd_replacement import CS
            H_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg[:3] == 'inp':
            from functions.svd_replacement import Inpainting
            if deg == 'inp_lolcat':
                loaded = np.load("inp_masks/lolcat_extra.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_ours':
                print("our inpainting mask\n\n")
                loaded = np.load("inp_masks/1.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(self.device).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'deno':
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride = factor)
        elif deg == 'deblur_uni':
            from functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'color':
            from functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'colorsrinpainting':
            from functions.svd_replacement import ColorSRInpainting
            H_funcs = ColorSRInpainting()
        else:
            print("ERROR: degradation type not supported")
            quit()
        args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y_0 = H_funcs.H(x_orig)
            
            b, hwc = y_0.size()
            if deg == 'color':
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y_0 = y_0.reshape((b, 1, h, w))
            elif "inp" in deg or "cs" in deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y_0 = y_0.reshape((b, 3, h, w))
            
            if args.noise_type == "gaussian":
                print("NOISE TYPE: GAUSSIAN")
                y_0 = get_gaussian_noisy_img(y_0, sigma_0)
            elif args.noise_type == "3d_gaussian":
                print("NOISE TYPE: 3D GAUSSIAN")
                y_0 = get_3d_gaussian_noisy_img(y_0, sigma_0)
            elif args.noise_type == "poisson":
                print("NOISE TYPE: POISSON")
                y_0 = get_poisson_noisy_img(y_0)
            elif args.noise_type == "speckle":
                print("NOISE TYPE: SPECKLE")
                y_0 = get_speckel_noisy_img(y_0, sigma_0)
            y_0 = y_0.reshape((b, hwc))

            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
            if deg[:6] == 'deblur': pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
            elif deg == 'color': pinv_y_0 = y_0.view(y_0.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1, 3, 1, 1)
            elif deg[:3] == 'inp': pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

            for i in range(len(pinv_y_0)):
                tvu.save_image(
                    inverse_data_transform(config, pinv_y_0[i]), os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]), os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
                )

            ##Begin DDIM
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            with torch.no_grad():
                x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)

            x = [inverse_data_transform(config, y) for y in x]

            for i in [-1]: #range(len(x)):
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                    )
                    if i == len(x)-1 or i == -1:
                        orig = inverse_data_transform(config, x_orig[j])
                        mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                        psnr = 10 * torch.log10(1 / mse)
                        avg_psnr += psnr

            idx_so_far += y_0.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn, classes=classes)
        if last:
            x = x[0][-1]
        return x

    def sample_ours(self, mode):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                # ckpt = '~/.cache/diffusion_models_converted/celeba_hq.ckpt'
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn

        self.sample_sequence_ours(model, cls_fn, mode)

    def sample_sequence_ours(self, model, cls_fn, mode):
        args, config = self.args, self.config

        # get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg[:3] == 'ccs':
            cs_ratio = int(deg[3:]) / 100.0
            from functions.svd_replacement import CS
            H_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg[:3] == 'inp':
            from functions.svd_replacement import Inpainting
            if deg == 'inp_lolcat':
#                 loaded = np.load("inp_masks/lolcat_extra.npy")
                loaded = np.load("inp_masks/mouth.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                missing_r = torch.randperm(config.data.image_size ** 2)[:config.data.image_size ** 2 // 2].to(
                    self.device).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'deno':
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0

            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif deg == 'deblur_uni':
            from functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'color':
            from functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'colorsrinpainting':
            from functions.svd_replacement import ColorSRInpainting
            H_funcs = ColorSRInpainting()
        elif deg[:2] == 'pd':
            from functions.svd_replacement import PD
            ratio = int(deg[2:])
            H_funcs = PD(config.data.channels, config.data.image_size, ratio)
        elif deg[:10] == 'maxpooling':
            from functions.svd_replacement import MaxPooling
            factor = int(deg[10:])
            H_funcs = MaxPooling(ratio=factor)
        elif deg == 'onebit':
            from functions.svd_replacement import OneBit
            H_funcs = OneBit()
        elif deg[:4] == 'jpeg':
            from functions.svd_replacement import JPEG
            qf = int(deg[4:])
            H_funcs = JPEG(config.data.channels, config.data.image_size, qf, self.device)
        elif deg[:5] == 'quant':
            from functions.svd_replacement import Quant
            qf = 256 // int(deg[5:])
            H_funcs = Quant(config.data.channels, config.data.image_size, qf, self.device)
        elif deg == 'sj':
            from functions.svd_replacement import SRJPEG
            H_funcs = SRJPEG(config.data.channels, config.data.image_size, self.device)
        elif deg == 'bs':
            from functions.svd_replacement import BlurSR
            H_funcs = BlurSR(config.data.channels, config.data.image_size, self.device)
        elif deg == 'bsj':
            from functions.svd_replacement import BlurSRJPEG
            H_funcs = BlurSRJPEG(config.data.channels, config.data.image_size, self.device)
        else:
            print("ERROR: degradation type not supported")
            quit()
        args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y_0 = H_funcs.H(x_orig)
            
            b, hwc = y_0.size()
            if 'color' in deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y_0 = y_0.reshape((b, 1, h, w))
            elif 'inp' in deg or 'cs' in deg or 'bit' in deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y_0 = y_0.reshape((b, 3, h, w))

            if args.noise_type == "gaussian":
                y_0 = get_gaussian_noisy_img(y_0, sigma_0)
            elif args.noise_type == "3d_gaussian":
                y_0 = get_3d_gaussian_noisy_img(y_0, sigma_0)
            elif args.noise_type == "poisson":
                y_0 = get_poisson_noisy_img(y_0)
            elif args.noise_type == "speckle":
                y_0 = get_speckel_noisy_img(y_0, sigma_0)
            y_0 = y_0.reshape((b, hwc))

            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size)
#             pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_size,
#                                                 self.config.data.image_size)
            if deg[:6] == 'deblur':
                pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_size,
                                    self.config.data.image_size)
            elif deg == 'color':
                pinv_y_0 = y_0.view(y_0.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      1)
            elif deg[:3] == 'inp':
                pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

            for i in range(len(pinv_y_0)):
                tvu.save_image(
                    inverse_data_transform(config, pinv_y_0[i]),
                    os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
                )

            ##Begin DDIM
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            has_grad = False
            with torch.no_grad():
                if mode == "algo1":
                    x, _ = self.sample_image_alogrithm_1(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, mu=args.mu)
                elif mode == "algo2":
                    x, _ = self.sample_image_alogrithm_2(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
                elif mode == "algo3":
                    x, _ = self.sample_image_alogrithm_3(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
                elif mode == "algo4":
                    x, _ = self.sample_image_alogrithm_4(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
                elif mode == "ddpm":
                    x, _ = self.sample_image_alogrithm_ddpm(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
                else:
                    has_grad = True
            
#             if mode == "algo1":
#                 x, _ = self.sample_image_alogrithm_1(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, mu=args.mu)
#             elif mode == "algo2":
#                 x, _ = self.sample_image_alogrithm_2(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
#             elif mode == "algo3":
#                 x, _ = self.sample_image_alogrithm_3(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
#             elif mode == "algo4":
#                 x, _ = self.sample_image_alogrithm_4(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
#             elif mode == "ddpm":
#                 x, _ = self.sample_image_alogrithm_ddpm(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
            if has_grad and mode == "grad":
                x, _ = self.sample_image_alogrithm_grad(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale)
            elif mode == "clip":
                x, _ = self.sample_image_alogrithm_clip(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale)
            elif has_grad and mode == "clip_ddim":
                x, _ = self.sample_image_alogrithm_clip_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, prompt=args.prompt, stop=args.stop)
            elif has_grad and mode == "clip_parser":
                x, _ = self.sample_image_alogrithm_clip_parser_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, prompt=args.prompt, stop=args.stop)
            elif has_grad and mode == "parse_clip":
                x, _ = self.sample_image_alogrithm_parse_clip_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, prompt=args.prompt, stop=args.stop, ref_path=args.ref_path)
            elif has_grad and mode == "clip_parser_edit":
                x, _ = self.sample_image_alogrithm_clip_parser_edit_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, prompt=args.prompt, stop=args.stop)
            elif has_grad and mode == "clip_edit":
                x, _ = self.sample_image_alogrithm_clip_edit_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, prompt=args.prompt, stop=args.stop)
            elif has_grad and mode == "parse_ddim":
                x, _ = self.sample_image_alogrithm_parse_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, stop=args.stop, ref_path=args.ref_path)
            elif has_grad and mode == "sketch_ddim":
                x, _ = self.sample_image_alogrithm_sketch_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, stop=args.stop, ref_path=args.ref_path)
            elif has_grad and mode == "land_ddim":
                x, _ = self.sample_image_alogrithm_land_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, stop=args.stop, ref_path=args.ref_path)
            elif has_grad and mode == "arc_ddim":
                x, _ = self.sample_image_alogrithm_arcface_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, stop=args.stop, ref_path=args.ref_path)
            elif has_grad and mode == "arcland_ddim":
                x, _ = self.sample_image_alogrithm_arcland_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, stop=args.stop, ref_path1=args.ref_path, ref_path2=args.ref_path2)
            elif has_grad and mode == "style_ddim":
                x, _ = self.sample_image_alogrithm_style_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, stop=args.stop, ref_path=args.ref_path)
            elif has_grad and mode == "style_transfer_ddim":
                x, _ = self.sample_image_alogrithm_style_transfer_ddim(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale, stop=args.stop, ref_path=args.ref_path)
            elif has_grad and mode == "gp":
                x, _ = self.sample_image_alogrithm_gp(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale)
            elif has_grad and mode == "gp_real_world":
                from functions.svd_replacement import SuperResolution
                H_funcs_sr = SuperResolution(config.data.channels, config.data.image_size, 4, self.device)
                x, _ = self.sample_image_alogrithm_gp(x, model, H_funcs, H_funcs_sr.H(x_orig), sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale)
            elif has_grad and mode == "ddgm_linear":
                x, _ = self.sample_image_alogrithm_ddgm_linear(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, rho_scale=args.rho_scale)
            elif has_grad and mode == "ilvr":
                x, _ = self.sample_image_alogrithm_ilvr(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)
            elif has_grad and mode == "ccdf":
                x, _ = self.sample_image_alogrithm_ccdf(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes, ccdf=args.ccdf)
            elif has_grad:
                x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)

            x = [inverse_data_transform(config, y) for y in x]

            for i in [-1]:  # range(len(x)):
#             for i in range(len(x)):
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                    )
                    if i == len(x) - 1 or i == -1:
                        orig = inverse_data_transform(config, x_orig[j])
                        mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                        psnr = 10 * torch.log10(1 / mse)
                        avg_psnr += psnr

            idx_so_far += y_0.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

    def sample_image_alogrithm_1(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, mu=0.02):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        k = 0.0005
        x = simple_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, k, cls_fn=cls_fn, classes=classes, mu=mu)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_2(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x = complex_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_3(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x = plus_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_4(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = grad_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_ddpm(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = ddpm_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_grad(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
#         x = grad_ddpm_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale)
        x = grad_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_clip(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = clip_ddpm_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_clip_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, prompt=None, stop=100):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = clip_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, prompt=prompt, stop=stop)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_clip_parser_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, prompt=None, stop=100):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = clip_parser_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, prompt=prompt, stop=stop)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_parse_clip_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, prompt=None, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = parse_clip_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, prompt=prompt, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_clip_parser_edit_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, prompt=None, stop=100):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = clip_parser_edit_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, prompt=prompt, stop=stop)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_clip_edit_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, prompt=None, stop=100):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = clip_edit_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, prompt=prompt, stop=stop)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_parse_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = parse_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_style_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = style_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_style_transfer_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = style_transfer_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_sketch_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = sketch_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_land_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = landmark_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_arcface_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = arcface_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_arcland_ddim(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0, stop=100, ref_path1=None, ref_path2=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = arcface_landmark_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale, stop=stop, ref_path1=ref_path1, ref_path2=ref_path2)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_gp(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = gp_ddim_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_ddgm_linear(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, rho_scale=1.0):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = ddgm_linear_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, rho_scale=rho_scale)
#         x = [x], []

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_ilvr(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = ilvr_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_ccdf(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None, ccdf=4):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = ccdf_diffusion(x, seq, model, self.betas, H_funcs, y_0, sigma_0, cls_fn=cls_fn, classes=classes, ccdf=ccdf)

        if last:
            x = x[0][-1]
        return x

