# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)

    def TorchRound(self):
        """
        Apply STE to clamp function.
        """
        class identity_quant(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                out = torch.round(input)
                return out

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        return identity_quant().apply

    def update_quantization_params(self, *args, **kwargs):
        self.q_max = self.observer.bit_type.upper_bound
        self.q_min = self.observer.bit_type.lower_bound
        self.eps = self.observer.eps
        self.max_val = nn.Parameter(self.observer.max_val, requires_grad=False)
        self.min_val = nn.Parameter(self.observer.min_val, requires_grad=False)

    def quant(self, inputs, scale=None, zero_point=None):
        scale = (self.max_val - self.min_val) / float(self.q_max - self.q_min)
        scale.clamp_(self.eps)
        zero_point = self.q_min - self.TorchRound()(self.min_val / scale)
        zero_point.clamp_(self.q_min, self.q_max)

        self.scale = scale
        self.zero_point = zero_point

        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs = self.TorchRound()(outputs).clamp(self.bit_type.lower_bound,
                                        self.bit_type.upper_bound)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs

