# Backbone networks used for face landmark detection
# Cunjian Chen (cunjian@msu.edu)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)
        
# USE global depthwise convolution layer. Compatible with MobileNetV2 (224×224), MobileNetV2_ExternalData (224×224)
class MobileNet_GDConv(nn.Module):
    def __init__(self,num_classes):
        super(MobileNet_GDConv,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)
    def forward(self,x):
        x = self.base_net(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


class LNDnet(nn.Module):
    def __init__(self, file_path):
        super(LNDnet, self).__init__()
        self.backbone = MobileNet_GDConv(136)
        state_dict = torch.load(file_path, map_location='cuda')['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module"):
                new_state_dict[key[7:]] = value
        self.backbone.load_state_dict(new_state_dict, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = torch.Tensor([ 0.229, 0.224, 0.225 ]).reshape(1, 3, 1, 1)
        
    def forward(self, x):
        self.backbone.eval()
        if x.shape[2] != 224:
            x = x[:, :, 16:240, 16:240]
        x_feats = self.backbone((((x + 1) / 2) - self.mean.to(x.device)) / self.std.to(x.device)).clamp(0, 1).reshape(-1, 68, 2)[:, 17:, :]
        return x_feats.reshape(x.size(0), -1)