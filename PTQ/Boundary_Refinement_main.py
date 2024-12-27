import os
import glob
import random
import argparse
import sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

from collections import OrderedDict
from basicsr.archs.craft_arch import CRAFT
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

def define_model(model_path, device):
    model = torch.load(model_path, weights_only=False)
    model = model.to(device)

    model.eval()

    return model, 255.

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
    
class PixelLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, loss_weight=1.0, reduction='mean', model_path=None, scale=2):
        super(PixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.window_size = 16
        self.scale = scale

        self.teacher = CRAFT(
                upscale=scale, img_size=(64, 64), window_size=self.window_size,
                img_range=1., depths=[2, 2, 2, 2],
                embed_dim=48, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                split_size_0=4,
                split_size_1=16
                )
        loadnet = torch.load(model_path)
        self.teacher.load_state_dict(loadnet['params'], strict=False)
        self.teacher.eval()

        self.criterion = torch.nn.MSELoss()

    def forward(self, x, pred):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained.
        """
        self.teacher.to(x.device)
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = x.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h_old + h_pad, :]
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w_old + w_pad]

            output = self.teacher(x)

            teacher_pred = output[..., :h_old * self.scale, :w_old * self.scale]

        return self.criterion(pred, teacher_pred)

def set_all_grads_enable(model):
    for name, param in model.named_parameters():
        if (name.endswith('max_val') or name.endswith('min_val')):
            param.requires_grad = True

def main():
    parser = argparse.ArgumentParser(description='PTQ-BR')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--output_dir', type=str, default='results/ptq')
    parser.add_argument('--saved_model_path', type=str, default='experiments/train_CRAFT_SR_X4/PTQ_models')
    parser.add_argument('--fp_model_path', type=str, default='experiments/train_CRAFT_SR_X4/float_models/craft_net_x4.pth')
    parser.add_argument('--ptq_adc_model_path', type=str, default='experiments/train_CRAFT_SR_X4/PTQ_models/craft_4bits_x4_ADC.pth')
    parser.add_argument('--traindir_LR', type=str, default='datasets/calibration_data/X4')
    parser.add_argument('--benchmarks', type=str, default='Set5+Set14+B100')
    parser.add_argument('--scale', default=4, type=int, help='scale')
    parser.add_argument('--epochs', default=10, type=int, help='total epochs')
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--bits', default=4, type=int, help='quantization bits')
    args = parser.parse_args()
    seed(args.seed)

    output_dir = args.output_dir
    bits = args.bits
    scale = args.scale
    traindir_LR = args.traindir_LR
    fp_model_path = args.fp_model_path
    ptq_adc_model_path = args.ptq_adc_model_path
    benchmarks = args.benchmarks.split('+')
    lr = args.lr
    total_epochs = args.epochs
    observe_str = 'dual'
    model_name = 'CRAFT'
    device = torch.device(args.device)
    model_folder_path = args.saved_model_path
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    window_size = 16

    model, norm = define_model(
                        model_path=ptq_adc_model_path, 
                        device=device
                    )

    # Set all gradients to false
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False

    model = model.to(device)
    model.train()
    
    # create dataloader
    lr_paths = sorted(glob.glob(os.path.join(traindir_LR, '*')))
    dataset = CaliDataset(lr_paths, norm)
    train_dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, pin_memory=True)

    # define loss
    ploss = PixelLoss(model_path=fp_model_path, scale=scale)

    set_all_grads_enable(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.model_quant()

    for _ in range(total_epochs):
        losses = []

        for idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            img_lq = data.to(device)
            
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq_new = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq_new = torch.cat([img_lq_new, torch.flip(img_lq_new, [3])], 3)[:, :, :, :w_old + w_pad]

            output = model(img_lq_new)
            output = output[..., :h_old * scale, :w_old * scale]

            loss = ploss(img_lq, output)
            losses.append(loss)
            
            loss.backward()
            optimizer.step()

        print('pixel loss:{:.4f}\tlr:{:.4f}'.format(sum(losses)/len(losses), optimizer.param_groups[0]['lr']))

    torch.save(model, '{}/{}_MODEL_{}bit_x{}.pth'.format(model_folder_path, model_name, bits, scale))

    model.eval()

    for benchmark in benchmarks:
        valdir_HR = 'datasets/benchmark/{}/HR'.format(benchmark)
        valdir_LR = 'datasets/benchmark/{}/LR_bicubic/X{}'.format(benchmark, scale)
        folder_path = '{}/{}/BR/{}bit/{}'.format(output_dir, observe_str, args.bits, benchmark)
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



if __name__ == '__main__':
    main()
