import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

from .clip.base_clip import CLIPEncoder
from .face_parsing.model import FaceParseTool
from .anime2sketch.model import FaceSketchTool 
from .landmark.model import FaceLandMarkTool
from .arcface.model import IDLoss


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def clip_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, prompt=None, stop=100, domain="face"):
    clip_encoder = CLIPEncoder().cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if domain == "face":
            repeat = 1
        elif domain == "imagenet":
            if 800 >= i >= 500:
                repeat = 10
            else:
                repeat = 1
        
        for idx in range(repeat):
        
            xt.requires_grad = True
            
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # get guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * 0.02
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            xt_next -= rho * norm_grad
            
            x0_t = x0_t.detach()
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def parse_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    parser = FaceParseTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = parser.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def sketch_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2sketch = FaceSketchTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            # print("use class_num")
            class_num = 7
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2sketch.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def landmark_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2landmark = FaceLandMarkTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2landmark.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def arcface_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    idloss = IDLoss(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = idloss.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

