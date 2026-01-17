import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable
from model.resnet_backbone import resnet50, oneByoneConvNet
import numpy as np


class CONV(nn.Module):
    def __init__(self, partials_num=5, ori_dim=64):
        super(CONV, self).__init__()
        self.partials_num = partials_num
        self.ori_dim = ori_dim
        self.conv0 = nn.Conv2d(ori_dim, ori_dim, kernel_size=1)
        # Create conv layers dynamically
        self.convs = nn.ModuleList()
        for i in range(partials_num):
            self.convs.append(nn.Conv2d(ori_dim, ori_dim, kernel_size=1))
    
    def forward(self, x_p, x):
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)
        
        x_sum = self.conv0(x)
        
        if self.partials_num > 0 and x_p is not None:
            x_p = F.interpolate(x_p, size=(h, w), mode='bilinear', align_corners=True)
            x_p = x_p.view(batch_size, self.partials_num, self.ori_dim, h, w)
            x_p_p = x_p.permute(1, 0, 2, 3, 4)
            
            for i in range(min(self.partials_num, len(self.convs))):
                x_sum += self.convs[i](x_p_p[i])
        
        return x_sum
class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1. / (1. + torch.exp(-10. * x))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
        x = torch.stack([unit.t()] * 16)
        y = torch.stack([unit] * 16)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()
        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0]*224, locs[i][1]*224, 44
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size-tl else in_size-tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size-tl else in_size-tl

            w_off = int(tx-tl) if (tx-tl) > 0 else 0
            h_off = int(ty-tl) if (ty-tl) > 0 else 0
            w_end = int(tx+tl) if (tx+tl) < in_size else in_size
            h_end = int(ty+tl) if (ty+tl) < in_size else in_size

            mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
            xatt = images[i] * mk
            xatt_cropped = xatt[:, w_off:w_end, h_off:h_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.interpolate(before_upsample, size=(88,88), mode='bilinear', align_corners = True)
            ret.append(xamp.data.squeeze())
        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        pass
class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """
    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)
class APN(nn.Module):
    def __init__(self,depth=18, partials_num = 5):
        super(APN, self).__init__()
        self.localization = LocalizationCNN(depth)
        self.partials_num = partials_num
        self.region_proposal = nn.Sequential(
        nn.Linear(2048 * 7 * 7, 1024),
        nn.Tanh(),
        nn.Linear(1024, 2*partials_num),
        nn.Sigmoid(),
        )
    def forward(self, x, x_ft):
        x_batch_size = x.size(0)
        x_l = self.localization(x,x_ft)
        x_l = x_l.view(-1, 2048*7*7)
        xs = self.region_proposal(x_l)
        xs_t = xs.view(x_batch_size * self.partials_num, 2)
        return xs_t
    
class LocalizationCNN(nn.Module):
    def __init__(self,depth = 18):
        super(LocalizationCNN, self).__init__()
        self.depth = depth
        self.conv1 = nn.Sequential( nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                    )
        self.conv2 = nn.Sequential( nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True)
                                    )
        self.conv3 = nn.Sequential( nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv4 = nn.Sequential( nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv5 = nn.Sequential( nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(2048),
                                    nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    def forward(self, x,x_ft):
        x = self.conv1(x) + x_ft[0].detach().float()
        x = self.conv2(x) + x_ft[1].detach().float()
        x = self.conv3(x) + x_ft[2].detach().float()
        x = self.conv4(x) + x_ft[3].detach().float()
        x = self.conv5(x) + x_ft[4].detach().float()
        return x
h = lambda x: 1. / (1. + torch.exp(-100. * x))
class APNResNet(nn.Module):
    def __init__(self, partials_num = 5, depth= 18):
        super(APNResNet, self).__init__()
        self.partials_num = partials_num
        
        self.whole_resnet = resnet50(pretrained=True)
        self.oconv0 = oneByoneConvNet(64,32,64)
        self.oconv1 = oneByoneConvNet(256,128,256)
        self.oconv2 = oneByoneConvNet(512,256,512)
        self.oconv3 = oneByoneConvNet(1024,512,1024)
        self.oconv4 = oneByoneConvNet(2048,1024,2048)

        self.attn_1 = CONV(partials_num=partials_num, ori_dim=64)
        self.attn_2 = CONV(partials_num=partials_num, ori_dim=256)
        self.attn_3 = CONV(partials_num=partials_num, ori_dim=512)
        self.attn_4 = CONV(partials_num=partials_num, ori_dim=1024)
        self.fn = nn.Linear(2048, 1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.APN = APN(depth, partials_num)
        self.crop_resize = AttentionCropLayer()

    def forward(self, x):
        batch_size = x.size(0)
        out1_w, out2_w, out3_w, out4_w, out5_w, out5_p_w = self.whole_resnet(x)

        out5_p_w = out5_p_w.view(batch_size, -1)
        
            
        xs_t=  self.APN(x.clone(), (out1_w, out2_w, out3_w,out4_w,out5_w))
        images = x
        images = images.repeat_interleave(self.partials_num, dim=0)
        scaled_x = self.crop_resize(images, xs_t)
        scaled_x_b = scaled_x.view(-1, 16, 88, 88)
        out1_p_b, out2_p_b, out3_p_b, out4_p_b, out5_p_b, out5_p_p_b = self.whole_resnet(scaled_x_b)
        whole_box_num = 5
        out1 = self.attention(out1_w, out1_p_b, 1, whole_box_num)
        out2 = self.attention(out2_w, out2_p_b, 2, whole_box_num)
        out3 = self.attention(out3_w, out3_p_b, 3, whole_box_num)
        out4 = self.attention(out4_w, out4_p_b, 4, whole_box_num)
        out5 = self.avgpool(self.oconv4(out5_p_b)).view(batch_size, self.partials_num, -1)

        out1 = self.oconv0(out1)
        out2 = self.oconv1(out2)
        out3 = self.oconv2(out3)
        out4 = self.oconv3(out4)
        out6 = self.fn(out5_p_w)
        

        
        return out1, out2, out3, out4, out5, out6, (xs_t, scaled_x)

    def attention(self, x, x_p, stage, box_num):
        if stage == 1:
            out = self.attn_1(x_p, x)
        if stage == 2:
            out = self.attn_2(x_p, x)
        if stage == 3:
            out = self.attn_3(x_p, x)
        if stage == 4:
            out = self.attn_4(x_p, x)
        return out
