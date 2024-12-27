import os
import glob
import random
import argparse
import sys
sys.path.append(os.path.abspath('.'))

import cv2
import torch
import numpy as np

from collections import OrderedDict
from basicsr.archs.craft_q_arch import Q_CRAFT
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import bgr2ycbcr
from basicsr.utils.Quantify.bit_type import *
from torch.utils import data as data
from torch.utils.data import DataLoader

def seed(seed=0):
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

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

class CaliDataset(data.Dataset):
    def __init__(self, train_LR_paths, norm=255) -> None:
        super().__init__()
        self.lr_paths = train_LR_paths
        self.norm = norm

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_path = self.lr_paths[idx]
        img_lq = cv2.imread(lr_path, cv2.IMREAD_COLOR).astype(np.float32) / self.norm
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float()  # CHW-RGB to NCHW-RGB
        return img_lq

def define_model(window_size, quantization_a, model_path, device, scale, bit_type_w, bit_type_a):
    model = Q_CRAFT(
        upscale=scale, img_size=(64, 64), window_size=window_size,
        img_range=1., depths=[2, 2, 2, 2],
        embed_dim=48, num_heads=[6, 6, 6, 6], mlp_ratio=2,
        split_size_0=4,
        split_size_1=16,
        quant=False,
        calibrate=False,
        bit_type_w=bit_type_w,
        bit_type_a=bit_type_a,
        observer_str_a=quantization_a,
        observer_str_w=quantization_a,
        calibration_mode_a='layer_wise',
        calibration_mode_w='channel_wise',
        quantizer_str='uniform',
    )

    loadnet = torch.load(model_path)

    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'

    model.load_state_dict(loadnet[keyname], strict=False)

    model = model.to(device)

    model.eval()

    return model, 255.

def main():
    parser = argparse.ArgumentParser(description='PTQ-ADC')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--output_dir', type=str, default='results/ptq')
    parser.add_argument('--saved_model_path', type=str, default='experiments/train_CRAFT_SR_X4/PTQ_models')
    parser.add_argument('--fp_model_path', type=str, default='experiments/train_CRAFT_SR_X4/float_models/craft_net_x4.pth')
    parser.add_argument('--traindir_LR', type=str, default='datasets/calibration_data/X4')
    parser.add_argument('--benchmarks', type=str, default='Set5+Set14+B100')
    parser.add_argument('--scale', default=4, type=int, help='scale')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--bits', default=4, type=int, help='quantization bits')

    args = parser.parse_args()
    seed(args.seed)

    output_dir = args.output_dir
    bits = args.bits
    scale = args.scale
    traindir_LR = args.traindir_LR
    fp_model_path = args.fp_model_path
    benchmarks = args.benchmarks.split('+')

    if bits == 4:
        activation = BIT_TYPE_DICT['uint4']
        weight = BIT_TYPE_DICT['uint4']
    elif bits == 6:
        activation = BIT_TYPE_DICT['uint6']
        weight = BIT_TYPE_DICT['uint6']
    elif bits == 8:
        activation = BIT_TYPE_DICT['uint8']
        weight = BIT_TYPE_DICT['uint8']

    model_folder_path = args.saved_model_path
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    observe_str = 'dual'
    model_name = 'craft'

    device = torch.device(args.device)

    print('Buid model...')
    print('Scale:{}\tModel:{}\tObserve type:{}\tBit:{}'.format(scale, model_name, observe_str, bits))

    window_size = 16
    model, norm = define_model(window_size=window_size, quantization_a=observe_str,
                        model_path=fp_model_path, 
                        device=device, 
                        scale=scale, 
                        bit_type_w=weight,
                        bit_type_a=activation,
                        )

    # create dataloader
    lr_paths = sorted(glob.glob(os.path.join(traindir_LR, '*')))
    dataset = CaliDataset(lr_paths, norm)
    train_dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, pin_memory=True)
    
    print('Calibrating...')
    model.model_open_calibrate()

    for idx, data in enumerate(train_dataloader):
        print('calibration id: {}'.format(idx))
        img_lq = data.to(device)
        if idx == len(train_dataloader)-1:
            model.model_open_last_calibrate()

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = model(img_lq)
            output = output[..., :h_old * scale, :w_old * scale]

    model.model_close_calibrate()
    
    # 保存模型
    torch.save(model, '{}/{}_{}bits_x{}_ADC.pth'.format(model_folder_path, model_name, bits, scale))

    print('Quantization...')
    model.model_quant()

    print('Evaluating...')
    for benchmark in benchmarks:
        valdir_HR = 'datasets/benchmark/{}/HR'.format(benchmark)
        valdir_LR = 'datasets/benchmark/{}/LR_bicubic/X{}'.format(benchmark, scale)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []
        test_results['psnr_b'] = []
        psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

        folder_path = '{}/{}/ADC/{}'.format(output_dir, observe_str, benchmark)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

                output = model(img_lq)
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
            print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(folder_path, ave_psnr, ave_ssim))
            if img_gt.ndim == 3:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))

if __name__ == '__main__':
    main()
