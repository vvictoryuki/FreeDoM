import torch
from torch import nn
from .facial_recognition.model_irse import Backbone
import torchvision


class IDLoss(nn.Module):
    def __init__(self, ref_path=None):
        super(IDLoss, self).__init__()
#         print('Loading ResNet ArcFace for ID Loss')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("/userhome/yjw/ddgm_exp/functions/arcface/model_ir_se50.pth"))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        self.to_tensor = torchvision.transforms.ToTensor()

        self.ref_path = "/userhome/yjw/ddgm_exp/functions/arcface/land.png" if not ref_path else ref_path
        from PIL import Image
        img = Image.open(self.ref_path)
        image = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        self.ref = img

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def get_residual(self, image):
        img_feat = self.extract_feats(image)
        ref_feat = self.extract_feats(self.ref)
        return ref_feat - img_feat







