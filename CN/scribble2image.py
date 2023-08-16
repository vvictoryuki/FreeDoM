from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument("--timesteps", type=int, default=100)
parser.add_argument("--prompt", type=str, default="young man, realistic photo")
parser.add_argument("--scribble_ref", type=str, default="./test_imgs/s5.png")
parser.add_argument("--style_ref", type=str, default="./test_imgs/xingkong.jpg")
parser.add_argument("--no_freedom", action="store_true")
args = parser.parse_args()


model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model, add_condition_mode="style", add_ref_path=args.style_ref, no_freedom=args.no_freedom)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    # with torch.no_grad():
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape

    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) < 127] = 255

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().detach().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


# input_image = cv2.imread("./test_imgs/s6.png")
# prompt = "airplane"
input_image = cv2.imread(args.scribble_ref)
prompt = args.prompt
a_prompt = ""
n_prompt = ""
num_samples = 1
image_resolution = 512
ddim_steps = args.timesteps
guess_mode = False
strength = 1.0
scale = 9.0
seed = args.seed
eta = 0.0
low_threshold = 100
high_threshold = 200

res = process(
    input_image=input_image, 
    prompt=prompt, 
    a_prompt=a_prompt, 
    n_prompt=n_prompt, 
    num_samples=num_samples, 
    image_resolution=image_resolution, 
    ddim_steps=ddim_steps, 
    guess_mode=guess_mode, 
    strength=strength, 
    scale=scale, 
    seed=seed, 
    eta=eta
)
count = 1
for img in res:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/scribble-style-{}.png".format(count), img)
    count += 1
