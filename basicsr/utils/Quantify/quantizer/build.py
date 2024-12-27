# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .uniform import UniformQuantizer

str2quantizer = {'uniform': UniformQuantizer}


def build_quantizer(quantizer_str, bit_type, observer, module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
