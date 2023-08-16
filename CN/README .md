# FreeDoM-CN-style/faceID

This repo is designed to add FreeDoM-based control to [ControlNet](https://github.com/lllyasviel/ControlNet). 

The support additional control include style and human face ID.

## Installation

#### Environment

You can directly use the conda environment for [ControlNet](https://github.com/lllyasviel/ControlNet). You can find configuration instructions in their GitHub links. Additionally, in order to run the face detection program, you also need to install `dlib`.

#### Pre-trained model

- ControlNet (download at [huggingface](https://huggingface.co/lllyasviel/ControlNet/tree/main/models))
  - place the scribble model at `./models/control_sd15_scribble.pth`
  - place the pose model at `./models/control_sd15_openpose.pth`

- [CLIP](https://github.com/openai/CLIP) Image Encoder for style condition
  - The model will automatically download.

- [ArcFace](https://arxiv.org/abs/1801.07698)
  - place the model at `./cldm/arcface/model_ir_se50.pth`

- Annotators used in ControlNet
  - the path is `./annotator/ckpts`, related models will automatically download.


## Quick Start

Just run `bash run.sh`, you will get the results.

The explanation of useful options:

- `--seed` you can choose your seeds to make results different!
- `--timesteps` the number of sampling times, we use 100 as default setting.
- `--prompt` is the text prompt for control
- `--pose_ref` `--id_ref` for openpose+faceID control
- `--scribble_ref` `--style_ref` for scribble+style control
- `--no_freedom` disable guidance based on FreeDoM.



