# -*- encoding: utf-8 -*-
import math

import torch
from torch import optim
# from UDL.Basis.criterion_metrics import *
# from UDL.pansharpening.common.evaluate import analysis_accu
# from UDL.Basis.module import PatchMergeModule
# from UDL.Basis.pytorch_msssim.cal_ssim import SSIM
from model import *
from torchsummary import summary
import torch.nn.functional as F


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def init_w(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # torch.nn.init.uniform_(m.weight, a=0, b=1)
            elif isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class SSDT(nn.Module):
    def __init__(self, args):
        super(SSDT, self).__init__()
        self.img_size = args.image_size
        print(self.img_size)
        self.in_channels = args.n_bands
        self.in_chans1 = args.n_bands + args.n_bands_rgb
        self.embed_dim = 72  # w-msa
        self.dim = 72  # w-xca
        self.t = Merge(img_size=self.img_size, patch_size=1, in_chans1=self.in_chans1, in_chans2=self.in_channels,
                       embed_dim=self.embed_dim, num_heads1=[9, 9, 9], window_size=8, group=8, mlp_ratio=4.,
                       dim=self.dim,
                       num_heads2=[8, 8, 8], ffn_expansion_factor=2.66, LayerNorm_type='WithBias', bias=False)
        self.visual_corresponding_name = {}
        init_w(self.t)
        self.visual_corresponding_name['hr'] = 'result'
        self.visual_names = ['hr', 'result']

    def forward(self, msi, hsi_u, hsi):
        '''
        :param pan:
        :param ms:
        :return:
        '''
        self.msi = msi
        self.hsi_u = hsi_u
        self.hsi = hsi
        xt = torch.cat((self.hsi_u, self.msi), 1)  # Bx34X64x64

        w_out = self.t(xt, self.hsi)
        self.result = w_out + self.hsi_u
        return self.result

    def name(self):
        return 'SSDT'

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion

