# FreeDoM (ICCV 2023)
[![arXiv](https://img.shields.io/badge/arXiv-<2303.09833>-<COLOR>.svg)](https://arxiv.org/abs/2303.09833)

The official implementation of the paper: 

["FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model"](https://arxiv.org/abs/2303.09833)

By [Jiwen Yu](https://scholar.google.com.hk/citations?user=uoRPLHIAAAAJ), [Yinhuai Wang](https://wyhuai.github.io/info/), [Chen Zhao](https://scholar.google.com/citations?user=dUWdX5EAAAAJ), [Bernard Ghanem](https://www.bernardghanem.com/), [Jian Zhang](https://jianzhang.tech/)

![](./figure/overview.png)

FreeDoM is a **simple but effective training-free** method generating results under control from various conditions using unconditional diffusion models. Specifically, we use off-the-shelf pre-trained networks to construct the time-independent energy function, which measures the distance between the given conditions and the intermediately generated images. Then we compute the energy gradient and use it to guide the generation process. FreeDoM **supports various conditions**, including texts, segmentation maps, sketches, landmarks, face IDs, and style images. FreeDoM **applies to different data domains**, including human faces, images from ImageNet, and latent codes. 

🎉🎉🎉 **_News (2023-07-14)_**: Congratulations on FreeDoM being accepted by ICCV 2023! Our open-source project is making progress, stay tuned for updates!

## Overall Experimental Configurations

| Model Source                                                 | Data Domain        | Resolution               | Original Conditions           | Additional Training-free Conditions                | Sampling Time*(s/image) |
| ------------------------------------------------------------ | ------------------ | ------------------------ | ----------------------------- | -------------------------------------------------- | ----------------------- |
| [SDEdit](https://github.com/ermongroup/SDEdit)               | aligned human face | $256\times256$           | None                          | parsing maps, sketches, landmarks, face IDs, texts | ≈20s            |
| [guided-diffusion](https://github.com/openai/guided-diffusion) | ImageNet           | $256\times256$           | None                          | texts, style images                                | ≈140s           |
| [guided-diffusion](https://github.com/openai/guided-diffusion) | ImageNet           | $256\times256$           | class label                   | style images                                       | ≈50s            |
| [Stable Diffusion](https://github.com/CompVis/stable-diffusion) | general images     | $512\times512$(standard) | texts                         | style images                                       | ≈84s            |
| [ControlNet](https://github.com/lllyasviel/ControlNet)       | general images     | $512\times512$(standard) | human poses, scribbles, texts | face IDs, style images                             | ≈120s           |

*The sampling time is tested on a GeForce RTX 3090 GPU card.

## Results

<details>
    <summary>Training-free <strong>style</strong> guidance + <strong>Stable Diffusion</strong> (click to expand) </summary>
    <img src = "./figure/SD_style.png" width=6000>
</details>

<details>
    <summary>Training-free <strong>style</strong> guidance + Scribble <strong>ControlNet</strong> (click to expand)</summary>
    <img src="./figure/CN_style.png" width=2000>
</details>

<details>
    <summary>Training-free <strong>face ID</strong> guidance + Human-pose <strong>ControlNet</strong> (click to expand)</summary>
    <img src="./figure/CN_id.png" width=2000>
</details>

<details>
    <summary>Training-free <strong>text</strong> guidance on <strong>human faces</strong> (click to expand)</summary>
    <img src="./figure/text_face.png" width=2000>
</details>

<details>
    <summary>Training-free <strong>segmentation</strong> guidance on <strong>human faces</strong> (click to expand)</summary>
    <img src="./figure/seg_face.png" width=2000>
</details>

<details>
    <summary>Training-free <strong>sketch</strong> guidance on <strong>human faces</strong> (click to expand)</summary>
    <img src="./figure/sketch_face.png" width=2000>
</details>

<details>
    <summary>Training-free <strong>landmarks</strong> guidance on <strong>human faces</strong> (click to expand)</summary>
    <img src="./figure/landmark_face.png" width=2000>
</details>

<details>
    <summary>Training-free <strong>face ID</strong> guidance on <strong>human faces</strong> (click to expand)</summary>
    <img src="./figure/id_face.png" width=2000>
</details>
<details>    <summary>Training-free <strong>face ID</strong> guidance + <strong>landmarks</strong> guidance on <strong>human faces</strong> (click to expand)</summary>
    <img src="./figure/land+id.png" width=2000>
</details>

<details>    <summary>Training-free <strong>text</strong> guidance + <strong>segmentation</strong> guidance on <strong>human faces</strong> (click to expand)</summary>
    <img src="./figure/seg+text.png" width=2000>
</details>

<details>
    <summary>Training-free <strong>style transferring</strong> guidance + <strong>Stable Diffusion</strong> (click to expand)</summary>
    <img src="./figure/SD_style_transfer.png" width=2000>
</details>

<details>
    <summary>Training-free <strong>text-guided</strong> face editting (click to expand)</summary>
    <img src="./figure/face_edit.png" width=2000>
</details>

## Acknowledgments

Our work is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:

- open-source pre-trained diffusion models:
  - (human face models) https://github.com/ermongroup/SDEdit
  - (ImageNet mdoels) https://github.com/openai/guided-diffusion
  - (Stable Diffusion) https://github.com/CompVis/stable-diffusion
  - (ControlNet) https://github.com/lllyasviel/ControlNet
- pre-trained networks for constructing the training-free energy functions:
  - (texts, style images) https://github.com/openai/CLIP
  - (face parsing maps) https://github.com/zllrunning/face-parsing.PyTorch
  - (sketches) https://github.com/Mukosame/Anime2Sketch
  - (face landmarks) https://github.com/cunjian/pytorch_face_landmark
  - (face IDs) [ArcFace(https://arxiv.org/abs/1801.07698)](https://arxiv.org/abs/1801.07698)
- time-travel strategy for better sampling:
  - (DDNM) https://github.com/wyhuai/DDNM
  - (Repaint) https://github.com/andreas128/RePaint

We also introduce some recent works that shared similar ideas by updating the clean intermediate results $\mathbf{x}_{0|t}$:

- concurrent conditional image generation methods:
  - https://github.com/arpitbansal297/Universal-Guided-Diffusion
  - https://github.com/pix2pixzero/pix2pix-zero
- zero-shot image restoration methods:
  - (DDNM) https://github.com/wyhuai/DDNM
  - (DDRM) https://github.com/bahjat-kawar/ddrm
  - (Repaint) https://github.com/andreas128/RePaint
  - (DPS) https://github.com/DPS2022/diffusion-posterior-sampling

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{yu2023freedom,
title={FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model},
author={Yu, Jiwen and Wang, Yinhuai and Zhao, Chen and Ghanem, Bernard and Zhang, Jian},
journal={arXiv:2303.09833},
year={2023}
}
```



