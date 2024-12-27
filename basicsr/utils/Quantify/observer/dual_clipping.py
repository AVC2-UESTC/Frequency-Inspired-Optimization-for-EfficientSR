import torch

from .base import BaseObserver


class DualObserver(BaseObserver):

    def __init__(self,
                 module_type,
                 bit_type,
                 calibration_mode, type='MAE'):
        super(DualObserver, self).__init__(module_type, bit_type,
                                                 calibration_mode)
        self.symmetric = self.bit_type.signed
        self.l_a = None
        self.r_a = None
        self.min_val = None
        self.max_val = None
        self.type = type

        self.init = 1e100
        

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.module_type == 'conv_weight':
            range_shape = (-1, 1, 1, 1)
        elif self.module_type == 'linear_weight':
            range_shape = (-1, 1)
        elif self.module_type == 'activation':
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape


    def simulated_quant(self, float_inp, max_val, min_val, qmax, qmin):
        def fft_image(image):
            magnitude_spectrum = torch.zeros((image.shape[1], image.shape[2])).to(image.device)
            for i in range(image.shape[0]):
                f = torch.fft.fft(image[i,:,:])
                magnitude_spectrum += torch.abs(f)
            return magnitude_spectrum
        scale = (max_val - min_val) / float(qmax - qmin)
        scale.clamp_(self.eps)
        zero_point = qmin - torch.round(min_val / scale)
        zero_point.clamp_(qmin, qmax)

        quantized_inp = (float_inp / scale + zero_point).round().clamp(qmin, qmax)
        dequantized_inp = (quantized_inp - zero_point) * scale

        if self.type == 'FGO':
            dequantized_inp_fft = fft_image(dequantized_inp.mean(1))
            float_fft = fft_image(float_inp.mean(1))
            score = (dequantized_inp_fft - float_fft).abs().mean()
            return score
        else:
            score = (dequantized_inp - float_inp).abs().mean()
        return score

    def update(self, float_inp, delta_step=1):
        float_inp_reshaped = self.reshape_tensor(float_inp)

        cur_max = float_inp_reshaped.max(axis=1).values
        cur_min = float_inp_reshaped.min(axis=1).values
        channels = float_inp_reshaped.shape[0]

        if self.calibration_mode == 'channel_wise':

            if self.max_val is None:
                self.max_val = cur_max

            if self.min_val is None:
                self.min_val = cur_min

            for c in range(channels):
                
                qmax = self.bit_type.upper_bound
                qmin = self.bit_type.lower_bound

                quant_step = self.bit_type.range

                edge_best_tmp_min = cur_min[c].cpu()
                edge_best_tmp_max = cur_max[c].cpu()
                self.l_a = self.min_val[c].cpu()
                self.r_a = self.max_val[c].cpu()

                temp = self.init

                bin_counts, bin_edges = torch.histogram(float_inp_reshaped[c:c+1].reshape(-1).cpu().clamp(edge_best_tmp_min, edge_best_tmp_max), bins=quant_step)

                l = 0
                r = len(bin_counts)

                while(1):

                    if l+delta_step == r or l == r-delta_step:
                        break

                    l_loss = self.simulated_quant(float_inp_reshaped[c:c+1], bin_edges[r], bin_edges[l + delta_step], qmax, qmin)
                    r_loss = self.simulated_quant(float_inp_reshaped[c:c+1], bin_edges[r - delta_step], bin_edges[l], qmax, qmin)

                    if l_loss < r_loss:
                        loss = l_loss
                        l = l + delta_step
                        max_val = bin_edges[r]
                        min_val = bin_edges[l]
                    else:
                        loss = r_loss
                        r = r - delta_step 
                        max_val = bin_edges[r]
                        min_val = bin_edges[l]

                    if loss > temp:
                        break
                    else:
                        temp = loss
                        edge_best_tmp_max = max_val
                        edge_best_tmp_min = min_val

                loss = temp
                gamma_left = 0.9
                gamma_right = 0.9

                self.l_a = gamma_left * self.l_a + (1-gamma_left) * edge_best_tmp_min
                self.r_a = gamma_right * self.r_a + (1-gamma_right) * edge_best_tmp_max

                self.max_val[c] = self.r_a.to(float_inp_reshaped.device)
                self.min_val[c] = self.l_a.to(float_inp_reshaped.device)

        else:

            qmax = self.bit_type.upper_bound
            qmin = self.bit_type.lower_bound

            min_val = cur_min.min()
            max_val = cur_max.max()

            if self.min_val is None:
                self.min_val = min_val
            if self.max_val is None:
                self.max_val = max_val

            quant_step = self.bit_type.range
            edge_best_tmp_min = min_val.cpu()
            edge_best_tmp_max = max_val.cpu()
            
            self.l_a = self.min_val
            self.r_a = self.max_val
            temp = self.init

            bin_counts, bin_edges = torch.histogram(float_inp_reshaped.reshape(-1).cpu().clamp(edge_best_tmp_min, edge_best_tmp_max), bins=quant_step)

            l = 0
            r = len(bin_counts)

            while(1):

                if l+delta_step >= r or l >= r-delta_step:
                    break

                l_loss = self.simulated_quant(float_inp, bin_edges[r], bin_edges[l + delta_step], qmax, qmin)
                r_loss = self.simulated_quant(float_inp, bin_edges[r - delta_step], bin_edges[l], qmax, qmin)

                if l_loss < r_loss:
                    loss = l_loss
                    l = l + delta_step
                    max_val = bin_edges[r]
                    min_val = bin_edges[l]
                else:
                    loss = r_loss
                    r = r - delta_step 
                    max_val = bin_edges[r]
                    min_val = bin_edges[l]

                if loss > temp:
                    break
                else:
                    temp = loss
                    edge_best_tmp_max = max_val
                    edge_best_tmp_min = min_val

            gamma_left = 0.9
            gamma_right = 0.9

            self.l_a = gamma_left * self.l_a + (1 - gamma_left) * edge_best_tmp_min
            self.r_a = gamma_right * self.r_a + (1 - gamma_right) * edge_best_tmp_max

            self.max_val = self.r_a.to(float_inp_reshaped.device)
            self.min_val = self.l_a.to(float_inp_reshaped.device)

    def get_quantization_params(self):

        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        if self.symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point
