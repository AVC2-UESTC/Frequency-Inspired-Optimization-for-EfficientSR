import sys 
import os
sys.path.append(os.path.abspath('./'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from basicsr.archs.arch_util import trunc_normal_
from einops import rearrange
from basicsr.utils.Quantify.layers import *
from basicsr.utils.Quantify.bit_type import *
# from basicsr.utils.registry import ARCH_REGISTRY


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(B, -1, H_sp* W_sp, C)

    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 quant=False,
                 calibrate=False,
                 bit_type_a=BIT_TYPE_DICT['int8'],
                 bit_type_w=BIT_TYPE_DICT['int8'],
                 calibration_mode_a='layer_wise',
                 calibration_mode_w='channel_wise',
                 observer_str_w='minmax',
                 observer_str_a='minmax',
                 quantizer_str='uniform'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = QLinear(in_features,
                           hidden_features,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=bit_type_w,
                           calibration_mode=calibration_mode_w,
                           observer_str=observer_str_w,
                           quantizer_str=quantizer_str)

        self.qact_fc1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)

        self.act = act_layer()

        self.qact_act = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)
        
        self.fc2 = QLinear(hidden_features,
                           out_features,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=bit_type_w,
                           calibration_mode=calibration_mode_w,
                           observer_str=observer_str_w,
                           quantizer_str=quantizer_str)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.qact_act(x)
        x = self.fc2(x)
        return x

class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads,
                residual,
                quant=False,
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['int8'],
                bit_type_w=BIT_TYPE_DICT['int8'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_w='minmax',
                observer_str_a='minmax',
                quantizer_str='uniform'):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj =  QLinear(2,
                                self.pos_dim,
                                quant=quant,
                                calibrate=calibrate,
                                bit_type=bit_type_w,
                                calibration_mode=calibration_mode_w,
                                observer_str=observer_str_w,
                                quantizer_str=quantizer_str)

        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            QAct(quant=quant,
                calibrate=calibrate,
                bit_type=bit_type_a,
                calibration_mode=calibration_mode_a,
                observer_str=observer_str_a,
                quantizer_str=quantizer_str),
            QLinear(self.pos_dim,
                    self.pos_dim,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=bit_type_w,
                    calibration_mode=calibration_mode_w,
                    observer_str=observer_str_w,
                    quantizer_str=quantizer_str)
        )
        
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            QAct(quant=quant,
                calibrate=calibrate,
                bit_type=bit_type_a,
                calibration_mode=calibration_mode_a,
                observer_str=observer_str_a,
                quantizer_str=quantizer_str),
            QLinear(self.pos_dim,
                    self.pos_dim,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=bit_type_w,
                    calibration_mode=calibration_mode_w,
                    observer_str=observer_str_w,
                    quantizer_str=quantizer_str)
        )
        
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            QAct(quant=quant,
                calibrate=calibrate,
                bit_type=bit_type_a,
                calibration_mode=calibration_mode_a,
                observer_str=observer_str_a,
                quantizer_str=quantizer_str),
            QLinear(self.pos_dim,
                    self.num_heads,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=bit_type_w,
                    calibration_mode=calibration_mode_w,
                    observer_str=observer_str_w,
                    quantizer_str=quantizer_str)
        )
        
    def forward(self, biases):
        self.l, self.c = biases.shape
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Attention_regular(nn.Module):
    """ Regular Rectangle-Window (regular-Rwin) self-attention with dynamic relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        resolution (int): Input resolution.
        idx (int): The identix of V-Rwin and H-Rwin, 0 is H-Rwin, 1 is Vs-Rwin. (different order from Attention_axial)
        split_size (tuple(int)): Height and Width of the regular rectangle window (regular-Rwin).
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """
    def __init__(self, dim, idx, split_size=[2,4], 
                dim_out=None, num_heads=6, 
                qk_scale=None, 
                position_bias=True,
                quant=False,
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['int8'],
                bit_type_w=BIT_TYPE_DICT['int8'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_w='minmax',
                observer_str_a='minmax',
                quantizer_str='uniform'):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp


        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False,
                 bit_type_a=bit_type_a,
                 bit_type_w=bit_type_w,
                 calibration_mode_a=calibration_mode_a,
                 calibration_mode_w=calibration_mode_w,
                 observer_str_a=observer_str_a,
                 observer_str_w=observer_str_w,
                 quantizer_str=quantizer_str)
        
        self.softmax = nn.Softmax(dim=-1)

        self.qact_q = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)
        
        self.qact_k = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)
        
        self.qact_v = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)
        
        self.qact_attn = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )


    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None, rpi=None, rpe_biases=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        self.N = L//(self.H_sp * self.W_sp)
        # partition the q,k,v, image to window

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        q = self.qact_q(q)
        k = self.qact_k(k)
        v = self.qact_v(v)


        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        pos = self.pos(rpe_biases)

        # select position bias
        relative_position_bias = pos[rpi.view(-1)].view(
            self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.qact_attn(attn)


        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x
    
class ShortCut(nn.Module):
    def __init__(self):
        super(ShortCut, self).__init__()

    def forward(self, input):
        return input


class SRWAB(nn.Module):
    r""" Shift Rectangle Window Attention Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        split_size (int): Define the window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 split_size=(2,2),
                 shift_size=(0,0),
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                quant=False,
                calibrate=False,
                bit_type_w=BIT_TYPE_DICT['int8'],
                bit_type_a=BIT_TYPE_DICT['int8'],
                calibration_mode_w='channel_wise',
                calibration_mode_a='layer_wise',
                observer_str_w='minmax',
                observer_str_a='minmax',
                quantizer_str='uniform'):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)

        self.short_cut = ShortCut()

        self.qkv = QLinear(dim,
                           dim*3,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=bit_type_w,
                           calibration_mode=calibration_mode_w,
                           observer_str=observer_str_w,
                           quantizer_str=quantizer_str)
        
        self.proj = QLinear(dim,
                           dim,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=bit_type_w,
                           calibration_mode=calibration_mode_w,
                           observer_str=observer_str_w,
                           quantizer_str=quantizer_str)


        self.branch_num = 2

        self.qact_v = QAct(quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_a,
                        calibration_mode=calibration_mode_a,
                        observer_str=observer_str_a,
                        quantizer_str=quantizer_str
                        )
        self.get_v = QConv2d(dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=3//2,
                groups=dim,
                quant=quant,
                calibrate=calibrate,
                bit_type=bit_type_w,
                calibration_mode=calibration_mode_w,
                observer_str=observer_str_w,
                quantizer_str=quantizer_str
                )

        self.attns = nn.ModuleList([
                Attention_regular(
                    dim//2, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, position_bias=True, 
                    quant=quant,
                    calibrate=calibrate,
                    bit_type_w=bit_type_w,
                    bit_type_a=bit_type_a,
                    calibration_mode_w=calibration_mode_w,
                    calibration_mode_a=calibration_mode_a,
                    observer_str_a=observer_str_a,
                    observer_str_w=observer_str_w,
                    quantizer_str=quantizer_str)
                for i in range(self.branch_num)])

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim,
                        hidden_features=mlp_hidden_dim,
                        act_layer=act_layer,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type_w=bit_type_w,
                        bit_type_a=bit_type_a,
                        calibration_mode_w=calibration_mode_w,
                        calibration_mode_a=calibration_mode_a,
                        observer_str_a=observer_str_a,
                        observer_str_w=observer_str_w,
                        quantizer_str=quantizer_str)
        
        self.qact_norm1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )

        self.qact_norm2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )
    

        self.qact_res1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )
        
        self.qact_v = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )
        

    def forward(self, x, x_size, params, attn_mask=NotImplementedError):
        h, w = x_size
        self.h,self.w = x_size

        b, l, c = x.shape
        shortcut = self.short_cut(x)

        x = self.norm1(x)
        x = self.qact_norm1(x)

        qkv = self.qkv(x).reshape(b, -1, 3, c).permute(2, 0, 1, 3) # 3, B, HW, C
        v = qkv[2].transpose(-2,-1).contiguous().view(b, c, h, w)

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            qkv = qkv.view(3, b, h, w, c)
            # H-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:c//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, b, h*w, c//2)
            # V-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,c//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, b, h*w, c//2)

            # H-Rwin
            x1_shift = self.attns[0](qkv_0, h, w, mask=attn_mask[0], rpi=params['rpi_sa_h'], rpe_biases=params['biases_h'])
            # V-Rwin
            x2_shift = self.attns[1](qkv_1, h, w, mask=attn_mask[1], rpi=params['rpi_sa_v'], rpe_biases=params['biases_v'])

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            # Concat
            attened_x = torch.cat([x1,x2], dim=-1)
        else:
            # H-Rwin
            x1 = self.attns[0](qkv[:,:,:,:c//2], h, w, rpi=params['rpi_sa_h'], rpe_biases=params['biases_h'])
            # V-Rwin
            x2 = self.attns[1](qkv[:,:,:,c//2:], h, w, rpi=params['rpi_sa_v'], rpe_biases=params['biases_v'])
            # Concat
            attened_x = torch.cat([x1,x2], dim=-1)

        attened_x = attened_x.view(b, -1, c).contiguous()

        # Locality Complementary Module
        lcm = self.get_v(self.qact_v(v))
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        attened_x = attened_x + lcm

        attened_x = self.qact_res1(attened_x)
        attened_x = self.proj(attened_x)

        # FFN
        x = shortcut + attened_x
        x = x + self.mlp(self.qact_norm2(self.norm2(x)))
        return x


class HFERB(nn.Module):
    def __init__(self, dim,
                quant=False, 
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['int8'],
                bit_type_w=BIT_TYPE_DICT['int8'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_a='minmax',
                observer_str_w='minmax',
                quantizer_str='uniform'):
        super().__init__()
        self.mid_dim = dim//2
        self.dim = dim
        self.act = nn.GELU()
        self.last_fc = QConv2d(self.dim,
                           self.dim,
                           kernel_size=1,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=bit_type_w,
                           calibration_mode=calibration_mode_w,
                           observer_str=observer_str_w,
                           quantizer_str=quantizer_str)

        # High-frequency enhancement branch
        self.fc = QConv2d(self.mid_dim,
                           self.mid_dim,
                           kernel_size=1,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=bit_type_w,
                           calibration_mode=calibration_mode_w,
                           observer_str=observer_str_w,
                           quantizer_str=quantizer_str)

        self.max_pool = nn.MaxPool2d(3, 1, 1)

        # Local feature extraction branch
        self.conv = QConv2d(self.mid_dim,
                           self.mid_dim,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=bit_type_w,
                           calibration_mode=calibration_mode_w,
                           observer_str=observer_str_w,
                           quantizer_str=quantizer_str)
        
        self.qact_last_fc = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str,
                          type='FGO')
        
        self.qact_lfe = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str,
                          type='FGO')
        
        self.qact_hfe = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str,
                          type='FGO')
        
        self.qact_maxpool = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str,
                          type='FGO'
                          )

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        short = x.clone()
        x_hfe = self.qact_hfe(x[:,self.mid_dim:,:,:])
        x_lfe = self.qact_lfe(x[:,:self.mid_dim,:,:])

        # Local feature extraction branch
        lfe = self.act(self.conv(x_lfe))

        # High-frequency enhancement branch
        hfe = self.act(self.fc(self.qact_maxpool(self.max_pool(x_hfe))))

        x = torch.cat([lfe, hfe], dim=1)

        x = self.qact_last_fc(x)
        x = self.last_fc(x)

        x = short + x
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim,
                num_heads, 
                bias, 
                quant=False, 
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['int8'],
                bit_type_w=BIT_TYPE_DICT['int8'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_w='minmax',
                observer_str_a='minmax',
                quantizer_str='uniform'):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dim = dim

        self.softmax = nn.Softmax(dim=-1)

        self.q = QConv2d(dim,
                        dim,
                        kernel_size=1,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        calibration_mode=calibration_mode_w,
                        observer_str=observer_str_w,
                        quantizer_str=quantizer_str)
        
        self.q_dwconv = QConv2d(dim,
                        dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=dim,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        calibration_mode=calibration_mode_w,
                        observer_str=observer_str_w,
                        quantizer_str=quantizer_str)
        
        self.kv = QConv2d(dim,
                        dim*2,
                        kernel_size=1,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        calibration_mode=calibration_mode_w,
                        observer_str=observer_str_w,
                        quantizer_str=quantizer_str)

        self.kv_dwconv = QConv2d(dim*2,
                        dim*2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=dim*2,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        calibration_mode=calibration_mode_w,
                        observer_str=observer_str_w,
                        quantizer_str=quantizer_str)
        
        self.project_out = QConv2d(dim,
                        dim,
                        kernel_size=1,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        calibration_mode=calibration_mode_w,
                        observer_str=observer_str_w,
                        quantizer_str=quantizer_str)
        
        self.qact_q = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)
        
        self.qact_k = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)
        
        self.qact_v = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)
        
        self.qact_attn = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )
        
        
        self.qact_conv_q = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )
        
        self.qact_conv_kv = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )
        
        self.qact_project = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str)

    def _forward(self, q, kv):
        k,v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)


        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q = self.qact_q(q)
        k = self.qact_k(k)
        v = self.qact_v(v)



        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = self.softmax(attn)
        attn = self.qact_attn(attn)

        out = (attn @ v)

        return out

    def forward(self, low, high):
        self.h, self.w = low.shape[2:]

        q = self.q_dwconv(self.qact_conv_q(self.q(high)))
        kv = self.kv_dwconv(self.qact_conv_kv(self.kv(low)))

        out = self._forward(q, kv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=kv.shape[-2], w=kv.shape[-1])
        out = self.qact_project(out)
        out = self.project_out(out)
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, 
                ffn_expansion_factor, 
                bias,
                quant=False, 
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['int8'],
                bit_type_w=BIT_TYPE_DICT['int8'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_a='minmax',
                observer_str_w='minmax',
                quantizer_str='uniform'):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.hid_fea = hidden_features
        self.dim = dim

        self.project_in = QConv2d(dim,
                        hidden_features*2,
                        kernel_size=1,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        calibration_mode=calibration_mode_w,
                        observer_str=observer_str_w,
                        quantizer_str=quantizer_str)

        self.qact_project_in = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )

        self.dwconv = QConv2d(hidden_features*2,
                        hidden_features*2,
                        kernel_size=3,
                        stride=1, 
                        padding=1,
                        groups=hidden_features*2,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        calibration_mode=calibration_mode_w,
                        observer_str=observer_str_w,
                        quantizer_str=quantizer_str)
        
        self.qact_dot = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )

        self.project_out = QConv2d(hidden_features,
                        dim,
                        kernel_size=1,
                        bias=bias,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_w,
                        calibration_mode=calibration_mode_w,
                        observer_str=observer_str_w,
                        quantizer_str=quantizer_str)
        
    def forward(self, x):
        self.h, self.w = x.shape[2:]
        x = self.project_in(x)
        x = self.qact_project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2 
        x = self.qact_dot(x)

        x = self.project_out(x)
        return x

##########################################################################
class HFB(nn.Module):
    r""" Hybrid Fusion Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        ffn_expansion_factor (int): Define the window size.
        bias (int): Shift size for SW-MSA.
        LayerNorm_type (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self, dim,
                num_heads,
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
                quant=False, 
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['int8'],
                bit_type_w=BIT_TYPE_DICT['int8'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_w='minmax',
                observer_str_a='minmax',
                quantizer_str='uniform'):
        super(HFB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        
        self.short_cut = ShortCut()
        
        self.attn = Attention(dim, 
                            num_heads, 
                            bias,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type_a=bit_type_a,
                            bit_type_w=bit_type_w,
                            calibration_mode_a=calibration_mode_a,
                            calibration_mode_w=calibration_mode_w,
                            observer_str_w=observer_str_w,
                            observer_str_a=observer_str_a,
                            quantizer_str=quantizer_str
                            )
        
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = FeedForward(dim,
                            ffn_expansion_factor,
                            bias,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type_a=bit_type_a,
                            bit_type_w=bit_type_w,
                            calibration_mode_a=calibration_mode_a,
                            calibration_mode_w=calibration_mode_w,
                            observer_str_w=observer_str_w,
                            observer_str_a=observer_str_a,
                            quantizer_str=quantizer_str
                            )
        self.dim = dim

        self.qact_norm1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )
        
        self.qact_norm2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )

        self.qact_high = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=bit_type_a,
                          calibration_mode=calibration_mode_a,
                          observer_str=observer_str_a,
                          quantizer_str=quantizer_str
                          )

    def forward(self, low, high):
        self.h, self.w = low.shape[2:]
        short_cut = self.short_cut(low)
        high = self.qact_high(high)
        x = short_cut + self.attn(self.qact_norm1(self.norm1(low)), high)
        x = x + self.ffn(self.qact_norm2(self.norm2(x)))

        return x

class CRFB(nn.Module):
    """ Cross-Refinement Fusion Block.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                dim,
                depth,
                num_heads,
                split_size_0=7,
                split_size_1=7,
                mlp_ratio=2.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                quant=False,
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['int4'],
                bit_type_w=BIT_TYPE_DICT['int4'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_a='minmax',
                observer_str_w='minmax',
                quantizer_str='uniform'):

        super().__init__()
        self.depth = depth

        # Shift Rectangle window attention blocks
        self.srwa_blocks = nn.ModuleList([
            SRWAB(
                dim=dim,
                num_heads=num_heads,
                split_size=[split_size_0,split_size_1],
                shift_size=[0,0] if (i % 2 == 0) else [split_size_0//2, split_size_1//2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                quant=quant,
                calibrate=calibrate,
                bit_type_a=bit_type_a,
                bit_type_w=bit_type_w,
                calibration_mode_a=calibration_mode_a,
                calibration_mode_w=calibration_mode_w,
                observer_str_a=observer_str_a,
                observer_str_w=observer_str_w,
                quantizer_str=quantizer_str)
                for i in range(2*depth)
        ])

        # High frequency enhancement residual blocks
        self.hfer_blocks = nn.ModuleList([
                HFERB(dim,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type_a=bit_type_a,
                    bit_type_w=bit_type_w,
                    calibration_mode_a=calibration_mode_a,
                    calibration_mode_w=calibration_mode_w,
                    observer_str_a=observer_str_a,
                    observer_str_w=observer_str_w,
                    quantizer_str=quantizer_str)
            for _ in range(depth)])

        # Hybrid fusion blocks
        self.hf_blocks = nn.ModuleList([
            HFB(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type='WithBias',
                quant=quant,
                calibrate=calibrate,
                bit_type_a=bit_type_a,
                bit_type_w=bit_type_w,
                calibration_mode_a=calibration_mode_a,
                calibration_mode_w=calibration_mode_w,
                observer_str_a=observer_str_a,
                observer_str_w=observer_str_w,
                quantizer_str=quantizer_str)
                for _ in range(depth)
            ])
    

    def forward(self, x, x_size, params):
        b, c, h, w = x.shape
        for i in range(self.depth):
            low = x.permute(0, 2, 3, 1)
            low = low.reshape(b, h*w, c)
            low = self.srwa_blocks[2*i+1](self.srwa_blocks[2*i](low, x_size, params, params['attn_mask']), x_size, params, params['attn_mask'])
            low = low.reshape(b, h, w, c)
            low = low.permute(0, 3, 1, 2)
            high = self.hfer_blocks[i](x)
            x = self.hf_blocks[i](low, high)
        return x



class RCRFG(nn.Module):
    """Residual Cross-Refinement Fusion Group (RCRFG).

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                dim,
                depth,
                num_heads,
                mlp_ratio=2.,
                qkv_bias=True,
                qk_scale=None,
                split_size_0 = 2,
                split_size_1 = 2,
                norm_layer=nn.LayerNorm, 
                quant=False,
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['int8'],
                bit_type_w=BIT_TYPE_DICT['int8'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_w='minmax',
                observer_str_a='minmax',
                quantizer_str='uniform'):
        super(RCRFG, self).__init__()

        self.dim = dim

        self.residual_group = CRFB(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            split_size_0 = split_size_0,
            split_size_1 = split_size_1,
            norm_layer=norm_layer,
            quant=quant,
            calibrate=calibrate,
            bit_type_a=bit_type_a,
            bit_type_w=bit_type_w,
            calibration_mode_a=calibration_mode_a,
            calibration_mode_w=calibration_mode_w,
            observer_str_a=observer_str_a,
            observer_str_w=observer_str_w,
            quantizer_str=quantizer_str)

        self.qact_input1 = QAct(quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type_a,
                        calibration_mode=calibration_mode_a,
                        observer_str=observer_str_a,
                        quantizer_str=quantizer_str
                        )
        
        self.conv = QConv2d(dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=3//2,
                quant=quant,
                calibrate=calibrate,
                bit_type=bit_type_w,
                calibration_mode=calibration_mode_w,
                observer_str=observer_str_w,
                quantizer_str=quantizer_str
                )

    def forward(self, x, x_size, params):
        self.h, self.w = x_size
        short_cut = x.clone()
        x = self.residual_group(x, x_size, params)
        x = self.conv(self.qact_input1(x))
        x = x + short_cut
        return x


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'
                 ):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        self.scale = scale
        m = []

        m.append(
                QConv2d(num_feat,
                        (scale ** 2) * num_out_ch,
                        kernel_size=3,
                        stride=1,
                        padding=3//2,
                        quant=quant,
                        calibrate=calibrate,
                        bit_type=bit_type,
                        calibration_mode=calibration_mode,
                        observer_str=observer_str,
                        quantizer_str=quantizer_str
                        )
            )
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


# @ARCH_REGISTRY.register()
class Q_CRAFT(nn.Module):
    r""" Cross-Refinement Adaptive Fusion Transformer
        Some codes are based on SwinIR.
    Args:
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        upscale: Upscale factor. 2/3/4/
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                in_chans=3,
                embed_dim=96,
                depths=(6, 6, 6, 6),
                num_heads=(6, 6, 6, 6),
                split_size_0 = 4,
                split_size_1 = 16,
                mlp_ratio=2.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                upscale=2,
                img_range=1.,
                upsampler='',
                resi_connection='1conv',
                quant=False,
                calibrate=False,
                bit_type_a=BIT_TYPE_DICT['uint8'],
                bit_type_w=BIT_TYPE_DICT['uint8'],
                calibration_mode_a='layer_wise',
                calibration_mode_w='channel_wise',
                observer_str_w='dual',
                observer_str_a='dual',
                quantizer_str='uniform',
                 **kwargs):
        super(Q_CRAFT, self).__init__()

        self.split_size = (split_size_0, split_size_1)

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.num_feat = num_feat
        self.num_out_ch = num_out_ch
        self.median_feas = []
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # relative position index
        self.calculate_rpi_v_sa()

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = QConv2d(num_in_ch,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=3//2,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=bit_type_w,
                    calibration_mode=calibration_mode_w,
                    observer_str=observer_str_w,
                    quantizer_str=quantizer_str
                    )
        
        self.qact_inpt1 = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=BIT_TYPE_DICT['uint8'],
                            calibration_mode=calibration_mode_a,
                            observer_str=observer_str_w,
                            quantizer_str=quantizer_str,
                            type='FGO')

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # build Residual Cross-Refinement Fusion Group
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RCRFG(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size_0 = split_size_0,
                split_size_1 = split_size_1,
                norm_layer=norm_layer,
                quant=quant,
                calibrate=calibrate,
                bit_type_a=bit_type_a,
                bit_type_w=bit_type_w,
                calibration_mode_a=calibration_mode_a,
                calibration_mode_w=calibration_mode_w,
                observer_str_w=observer_str_w,
                observer_str_a=observer_str_a,
                quantizer_str=quantizer_str)
            self.layers.append(layer)
            
        self.norm = LayerNorm(self.num_features, 'with_bias')

        self.qact_inpt3 = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=bit_type_a,
                            calibration_mode=calibration_mode_a,
                            observer_str=observer_str_a,
                            quantizer_str=quantizer_str,
                            type='FGO')
        
        self.qact_inpt4 = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=BIT_TYPE_DICT['uint8'],
                            calibration_mode=calibration_mode_a,
                            observer_str=observer_str_a,
                            quantizer_str=quantizer_str,
                            type='FGO')

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = QConv2d(embed_dim,
                            embed_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=bit_type_w,
                            calibration_mode=calibration_mode_w,
                            observer_str=observer_str_w,
                            quantizer_str=quantizer_str)
            self.qact_inpt2 = QAct(quant=quant,
                                calibrate=calibrate,
                                bit_type=bit_type_a,
                                calibration_mode=calibration_mode_a,
                                observer_str=observer_str_a,
                                quantizer_str=quantizer_str,
                                type='FGO')
            
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                quant=quant,
                                calibrate=calibrate,
                                bit_type=bit_type_w,
                                calibration_mode=calibration_mode_w,
                                observer_str=observer_str_w,
                                quantizer_str=quantizer_str
                                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_v_sa(self):
        # generate mother-set
        H_sp, W_sp = self.split_size[0], self.split_size[1]
        position_bias_h = torch.arange(1 - H_sp, H_sp)
        position_bias_w = torch.arange(1 - W_sp, W_sp)
        biases_h = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
        biases_h = biases_h.flatten(1).transpose(0, 1).contiguous().float()

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(H_sp)
        coords_w = torch.arange(W_sp)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += H_sp - 1
        relative_coords[:, :, 1] += W_sp - 1
        relative_coords[:, :, 0] *= 2 * W_sp - 1
        relative_position_index_h = relative_coords.sum(-1)


        H_sp, W_sp = self.split_size[1], self.split_size[0]
        position_bias_h = torch.arange(1 - H_sp, H_sp)
        position_bias_w = torch.arange(1 - W_sp, W_sp)
        biases_v = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
        biases_v = biases_v.flatten(1).transpose(0, 1).contiguous().float()

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(H_sp)
        coords_w = torch.arange(W_sp)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += H_sp - 1
        relative_coords[:, :, 1] += W_sp - 1
        relative_coords[:, :, 0] *= 2 * W_sp - 1
        relative_position_index_v = relative_coords.sum(-1)
        self.register_buffer('relative_position_index_h', relative_position_index_h)
        self.register_buffer('relative_position_index_v', relative_position_index_v)
        self.register_buffer('biases_v', biases_v)
        self.register_buffer('biases_h', biases_h)

        return biases_v, biases_h

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    
    def model_quant(self):
            for m in self.modules():
                if type(m) in [QConv2d, QLinear, QAct]:
                    m.quant = True

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct]:
                m.calibrate = False

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        params = {'attn_mask': (None, None), 'rpi_sa_h': self.relative_position_index_h, 'rpi_sa_v': self.relative_position_index_v, 'biases_v':self.biases_v, 'biases_h':self.biases_h}

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)

        return x

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.qact_inpt1(x)
        x = self.conv_first(x)
        short_cut = x.clone()
        x = self.forward_features(x)
        x = self.qact_inpt2(x)
        x = self.conv_after_body(x)
        x = x + short_cut
        x = self.qact_inpt3(x)
        x = self.upsample(x)
        x = self.qact_inpt4(x)
        x = x / self.img_range + self.mean
        return x

if __name__ == '__main__':
    import sys 
    import os
    sys.path.append(os.path.abspath('.'))
    upscale = 4
    window_size = 16
    height = (256 // upscale // window_size) * window_size
    width = (256 // upscale // window_size) * window_size
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = Q_CRAFT(
        upscale=upscale, img_size=(64, 64), window_size=window_size,
        img_range=1., depths=[2, 2, 2, 2],
        embed_dim=48, num_heads=[6, 6, 6, 6], mlp_ratio=2,
        split_size_0=4,
        split_size_1=16,
        quant=False,
        calibrate=False,
        calibration_mode_a='layer_wise',
        calibration_mode_w='channel_wise',
        quantizer_str='uniform',
    ).cuda()

    params = sum(map(lambda x: x.numel(), model.parameters()))
    results = dict()
    results[f"runtime"] = []
    model.eval()

    print('Calibrating...')
    model.model_open_calibrate()
    x = torch.randn((1, 3, height, width)).cuda()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i in range(10):
            print(i)
            x_sr = model(x)
        model.model_open_last_calibrate()
        x_sr = model(x)
        model.model_close_calibrate()
        model.model_quant()
        for _ in range(100):
            start.record()
            x_sr = model(x)
            end.record()
            torch.cuda.synchronize()
            results[f"runtime"].append(start.elapsed_time(end))  # milliseconds
    print(x.shape)

    print("{:.2f}ms".format(sum(results[f"runtime"]) / len(results[f"runtime"])))
    results["memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    print("Max Memery:{:.2f}[M]".format(results["memory"]))
    print("Height:{}->{}\nWidth:{}->{}\nParameters:{:.2f}K".format(height, x_sr.shape[2], width, x_sr.shape[3], params / 1e3))