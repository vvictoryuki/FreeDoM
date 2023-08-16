# FreeDoM-Face/ImageNet

This repo is designed to add FreeDoM-based control to diffusion models in the domain of face or ImageNet. 

The pretrained diffusion models include human face model provided by [SDEdit](https://github.com/ermongroup/SDEdit) and unconditional [guided-diffusion](https://github.com/openai/guided-diffusion), the support image resolution is $256\times 256$.

## Installation

#### Environment

You can directly use the conda environment for [Stable Diffusion](https://github.com/CompVis/stable-diffusion) or [DDNM](https://github.com/wyhuai/DDNM). You can find configuration instructions in their GitHub links.

#### Pre-trained model

- human face diffusion model provided by [SDEdit](https://github.com/ermongroup/SDEdit)
  - place the model in this directory `./exp/logs/celeba/celeba_hq.ckpt`
- unconditional [guided diffusion model](https://github.com/openai/guided-diffusion)
  - place the model in this directory `./exp/logs/imagenet/256x256_diffusion_uncond.pt`
- [CLIP](https://github.com/openai/CLIP)
  - The model will automatically download.
- [face parsing model](https://github.com/zllrunning/face-parsing.PyTorch)
  - place the model in this directory `./functions/face_parsing/79999_iter.pth`
- [sketch model](https://github.com/Mukosame/Anime2Sketch)
  - place the model in this directory `./functions/anime2sketch/netG.pth`
- [landmark model](https://github.com/cunjian/pytorch_face_landmark)
  - place the model in this directory:
    -  `./functions/landmark/checkpoint/mobilefacenet_model_best.pth.tar`
    - `./functions/landmark/checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar`
    - `./functions/landmark/Retinaface/weights/mobilenet0.25_Final.pth`
- [ArcFace model](https://arxiv.org/abs/1801.07698)
  - place the model in this directory `./functions/arcface/model_ir_se50.pth`

## Quick Start

Just run `bash run.sh`, you will get the results.

The explanation of useful options:

- `-i` is the folder name of the results.
- `-s` is the sampling method with different conditions, we support `	clip_ddim | parse_ddim | sketch_ddim | land_ddim | arc_ddim    `
- `--doc` for human face model, choose `celeba_hq`; for ImageNet model, choose `imagenet`
- `--timesteps` the number of sampling times, we use 100 as default setting.
- `--seed` you can choose your seeds to make results different!
  - `--model_type` for human face model, choose `"face"`; for ImageNet model, choose `"imagenet"`
- `--prompt` is the text prompt for CLIP condition control
- `--batch_size` control the number of generated images
- `--ref_path` is the path of reference image



