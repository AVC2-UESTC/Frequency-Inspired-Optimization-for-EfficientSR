import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.Quantify.layers import *
from basicsr.utils.Quantify.bit_type import *

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def quant_conv(in_channels, out_channels, kernel_size, bias=True,
                 quant=False,
                 calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'
                ):
    return QConv2d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=bias,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=bit_type,
                    calibration_mode=calibration_mode,
                    observer_str=observer_str,
                    quantizer_str=quantizer_str
                    )

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,
        quant=False,
        calibrate=False,
        last_calibrate=False,
        bit_type_w=BIT_TYPE_DICT['int6'],
        bit_type_a=BIT_TYPE_DICT['uint6'],
        observer_str_a='minmax',
        observer_str_w='minmax',
        calibration_mode_a='layer_wise',
        calibration_mode_w='channel_wise',
        quantizer_str='uniform'
          ):

        super(ResBlock, self).__init__()

        self.quan_a1 = QAct(quant=quant,
                    calibrate=calibrate,
                    bit_type=bit_type_a,
                    calibration_mode=calibration_mode_a,
                    observer_str=observer_str_a,
                    quantizer_str=quantizer_str)

        self.quan_a2 = QAct(quant=quant,
                    calibrate=calibrate,
                    bit_type=bit_type_a,
                    calibration_mode=calibration_mode_a,
                    observer_str=observer_str_a,
                    quantizer_str=quantizer_str)
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats,
                        kernel_size,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        observer_str=observer_str_w,
                        calibration_mode=calibration_mode_w,
                        quantizer_str=quantizer_str
                        ))
            # m.append(conv(n_feats, n_feats,
            #             kernel_size,
            #             bias=bias
            #             ))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = x.clone()
        x = self.quan_a1(x)
        x = self.body[0](x)
        x = self.body[1](x)
        x = self.quan_a2(x)
        x = self.body[2](x).mul(self.res_scale)
        x += res

        return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

