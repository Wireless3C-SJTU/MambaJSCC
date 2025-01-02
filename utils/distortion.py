import torch
import numpy as np
from taming.modules.losses.vqperceptual import * 
from taming.modules.losses.lpips import LPIPS as lpips
@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


@torch.jit.script
def _gaussian_filter(x, window_1d, use_padding: bool):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out


@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs


@torch.jit.script
def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False, eps: float = 1e-8):
    '''
    interface of ms-ssim
    :param X: a batch of images, (N,C,H,W)
    :param Y: a batch of images, (N,C,H,W)
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param weights: weights for different levels
    :param use_padding: padding image before conv
    :param eps: use for avoid grad nan.
    :return:
    '''
    weights = weights[:, None]

    levels = weights.shape[0]
    vals = []
    for i in range(levels):
        ss, cs = ssim(X, Y, window=window, data_range=data_range, use_padding=use_padding)

        if i < levels - 1:
            vals.append(cs)
            X = F.avg_pool2d(X, kernel_size=2, stride=2, ceil_mode=True)
            Y = F.avg_pool2d(Y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            vals.append(ss)

    vals = torch.stack(vals, dim=0)
    # Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
    vals = vals.clamp_min(eps)
    # The origin ms-ssim op.
    ms_ssim_val = torch.prod(vals[:-1] ** weights[:-1] * vals[-1:] ** weights[-1:], dim=0)
    # The new ms-ssim op. But I don't know which is best.
    # ms_ssim_val = torch.prod(vals ** weights, dim=0)
    # In this file's image training demo. I feel the old ms-ssim more better. So I keep use old ms-ssim op.
    return ms_ssim_val


class SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding

    @torch.jit.script_method
    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return r[0]


class MS_SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding', 'eps']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=1.0, channel=3, use_padding=False, weights=None,
                 levels=None, eps=1e-8):
        """
        class for ms-ssim
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels
        :param use_padding: padding image before conv
        :param weights: weights for different levels. (default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        :param levels: number of downsampling
        :param eps: Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
        """
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer('weights', weights)

    @torch.jit.script_method
    def forward(self, X, Y):
        return 1 - ms_ssim(X, Y, window=self.window, data_range=self.data_range, weights=self.weights,
                       use_padding=self.use_padding, eps=self.eps)

class loss_matrix(torch.nn.Module):
    def __init__(self,config):
        super(loss_matrix, self).__init__()
        self.config=config
        self.Cal_lpips=lpips().eval().cuda()
        if self.config.TRAIN.GAN_LOSS:
            self.discriminator = NLayerDiscriminator(input_nc=config.MODEL.VSSM.OUT_CHANS,
                                            n_layers=config.MODEL.disc_num_layers,
                                            use_actnorm=config.MODEL.use_actnorm
                                            ).cuda().apply(weights_init)
            self.discriminator_weight=config.TRAIN.DIS_WEIGHT
            self.disc_loss=hinge_d_loss
        _loss_dict=dict(
            PSNR=self.MSE_loss,
            MSSSIM=self.MSSSIM_loss,
            LPIPS=self.LPIPS_loss
        )
        self.loss=_loss_dict.get(config.TRAIN.LOSS, None)

    def MSSSIM_loss(self,x,y):
        CalcuSSIM=MS_SSIM(data_range=1., levels=4, channel=3).cuda()
        rec_loss = CalcuSSIM(x, y).mean() * x.numel() / x.shape[0]

        return rec_loss
    
    def MSE_loss(self,x,y):
        
        rec_loss = torch.nn.functional.mse_loss(x,y,reduction='sum') / x.shape[0]

        return rec_loss
    
    def LPIPS_loss(self,x,y):
        
        rec_loss = self.Cal_lpips.forward(x,y).mean() * x.numel() / x.shape[0]
        #print(rec_loss)
        return rec_loss
    
    def forward(self, recon, input, feature, last_layer=None, opt_idx=0, global_step=0):
        if self.config.TRAIN.GAN_LOSS:
            if opt_idx==0:  ## update autoencoder
            # reconstruction loss

                recon_loss=self.loss(recon, input, reduction='sum')/input.shape[0]
                # GAN loss

                logits_fake=self.discriminator(feature)
                g_loss = -torch.mean(logits_fake)
                d_weight=self.calculate_adaptive_weight(recon_loss, g_loss, last_layer=last_layer) if last_layer is not None else 0.5
                if global_step+1>self.config.TRAIN.START_EPOCH:
                    g_factor=1
                else:
                    g_factor=0
                loss=recon_loss+d_weight*g_factor*g_loss
                #print(recon_loss,d_weight*g_loss)
                #print(loss)
                return loss
            
            if opt_idx==1:
                Gaussian= torch.normal(mean=0.0, std=1, size=feature.shape).cuda()
                #print(Gaussian.shape)
                logits_fake=self.discriminator(feature)
                logits_real=self.discriminator(Gaussian)
                loss=self.disc_loss(logits_real, logits_fake)
                #print(torch.mean(logits_fake),torch.mean(logits_real))
                return loss
        else:

            return self.loss(recon, input)

        
        return 

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
class eval_matrix(torch.nn.Module):
    def __init__(self,config):
        super(eval_matrix, self).__init__()
        self.Cal_lpips=lpips().eval().cuda()
        _loss_dict=dict(
            PSNR=self.psnr,
            MSSSIM=self.msssim,
            LPIPS=self.LPIPS_loss,
        )
        self.loss=_loss_dict.get(config.TRAIN.EVAL_MATRIX, None)

    def psnr(self,x,y):
        #print(x.shape,y.shape)
        mse=torch.nn.MSELoss()(x.clamp(0.,1.)*255., y.clamp(0.,1.)*255.)
        #loss=torch.nn.MSELoss(x.clamp(0.,1.), y.clamp(0.,1.))
        psnr=10 * (torch.log(255. * 255. / mse) / np.log(10)).item()
        return psnr
    
    def msssim(self,x,y):
        CalMSSSIM=MS_SSIM(data_range=1., levels=4, channel=3).cuda()
        msssim=1-CalMSSSIM(x, y).mean().item()
        return msssim
    
    def LPIPS_loss(self,x,y):
       
        rec_loss = self.Cal_lpips(x,y).mean().item()
        return rec_loss
    
    def forward(self, x, y):
        return self.loss(x, y)
        