import torch
import torch.nn as nn
from .models.mobilefacenet import MobileFaceNet
import glob
import cv2
from PIL import Image
from .Retinaface import Retinaface
from .common.utils import BBox


class FaceLandMarkTool(nn.Module):
    def __init__(self, ref_path=None):
        super(FaceLandMarkTool, self).__init__()
        self.out_size = 112
        map_location = lambda storage, loc: storage.cuda()
        self.landmark_net = MobileFaceNet([self.out_size, self.out_size], 136)
        checkpoint = torch.load('/userhome/yjw/ddgm_exp/functions/landmark/checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)
        self.landmark_net.load_state_dict(checkpoint['state_dict'])
        self.landmark_net = self.landmark_net.eval()
        
        self.ref_path = "/userhome/yjw/ddgm_exp/functions/landmark/3650.png" if not ref_path else ref_path
        img = cv2.imread(self.ref_path)
        retinaface = Retinaface.Retinaface()
        faces = retinaface(img)
        face = faces[0]
        
        x1 = face[0]
        y1 = face[1]
        x2 = face[2]
        y2 = face[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h])*1.2)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        
        self.top, self.bottom, self.left, self.right = new_bbox.top, new_bbox.bottom, new_bbox.left, new_bbox.right
        
        cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
        cropped_face = cv2.resize(cropped, (self.out_size, self.out_size))
        
        test_face = cropped_face.copy()
        test_face = test_face/255.0
        test_face = test_face.transpose((2, 0, 1))
        test_face = test_face.reshape((1,) + test_face.shape)

        input_ref = torch.from_numpy(test_face).float()
        input_ref = torch.autograd.Variable(input_ref)

        self.landmark_ref = self.landmark_net(input_ref)[0].cuda()
        
    def get_residual(self, image):
        image = (image + 1.0) / 2.0
        image = image[:, :, self.top:self.bottom, self.left:self.right]
        image = torch.nn.functional.interpolate(image, size=self.out_size, mode='bicubic')
        landmark_img = self.landmark_net(image)[0]
        return self.landmark_ref - landmark_img
        
        
        
        
        
