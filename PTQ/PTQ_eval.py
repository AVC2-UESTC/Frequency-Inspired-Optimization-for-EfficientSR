import argparse
import random
import glob
import os
import sys
sys.path.append(os.path.abspath('.'))

import cv2
import torch
import numpy as np

from collections import OrderedDict
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import bgr2ycbcr
from basicsr.utils.Quantify.bit_type import *
from torch.utils import data as data

def get_image_pair(path, folder_lq=None, scale=4, tail=False, norm=255):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / norm
    if tail:
        img_lq = cv2.imread(f'{folder_lq}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(
            np.float32) / norm
    else:
        img_lq = cv2.imread(f'{folder_lq}/{imgname}x{scale}{imgext}', cv2.IMREAD_COLOR).astype(
            np.float32) / norm

    return imgname, img_lq, img_gt


parser = argparse.ArgumentParser(description='PTQ-CRAFT')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--output_dir', type=str, default='results/ptq')
parser.add_argument('--model_path', type=str, default='experiments/train_CRAFT_SR_X4/PTQ_models/craft_6bit_x4.pth')
parser.add_argument('--benchmarks', type=str, default='Set5+Set14+B100')
parser.add_argument('--scale', default=4, type=int, help='scale')
parser.add_argument('--bits', default=4, type=int, help='quantization bits')

args = parser.parse_args()

output_dir = args.output_dir
scale = args.scale
device = args.device
model_path = args.model_path
benchmarks = args.benchmarks.split('+')

observe_str = 'dual'
quantization_a = 'dual'
model_name = 'CRAFT'
window_size = 16
norm = 255

model = torch.load(model_path, weights_only=False)
model = model.to(device)

model.model_quant()
model.eval()

for benchmark in benchmarks:
    valdir_HR = 'datasets/benchmark/{}/HR'.format(benchmark)
    valdir_LR = 'datasets/benchmark/{}/LR_bicubic/X{}'.format(benchmark, scale)
    folder_path = '{}/{}/eval/{}bit/{}'.format(output_dir, observe_str, args.bits, benchmark)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(valdir_HR, '*')))):
        imgname, img_lq, img_gt = get_image_pair(path, valdir_LR, scale, False, norm=norm)  # image to HWC-BGR, float32
        imgname = os.path.splitext(os.path.basename(path))[0]
        # read image
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq_new = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq_new = torch.cat([img_lq_new, torch.flip(img_lq_new, [3])], 3)[:, :, :, :w_old + w_pad]

            output = model(img_lq_new)
            output = output[..., :h_old * scale, :w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 255/norm).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * norm).round().astype(np.uint8)
        cv2.imwrite(os.path.join(folder_path, f'{imgname}_{observe_str}_{model_name}.png'), output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * norm).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * scale, :w_old * scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = calculate_psnr(output, img_gt, crop_border=scale)
            ssim = calculate_ssim(output, img_gt, crop_border=scale)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                output_y = bgr2ycbcr(output.astype(np.float32) / norm) * norm
                img_gt_y = bgr2ycbcr(img_gt.astype(np.float32) / norm) * norm
                psnr_y = calculate_psnr(output_y, img_gt_y, crop_border=scale, test_y_channel=True)
                ssim_y = calculate_ssim(output_y, img_gt_y, crop_border=scale, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
                'PSNR_B: {:.2f} dB.'.
                format(idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format('N/A', ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))


