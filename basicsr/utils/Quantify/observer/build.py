# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .dual_clipping import DualObserver


str2observer = {
    'dual': DualObserver,
}


def build_observer(observer_str, module_type, bit_type, calibration_mode, type='MAE'):
    observer = str2observer[observer_str]
    return observer(module_type, bit_type, calibration_mode, type=type)

