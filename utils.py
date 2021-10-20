import torch
import torchgeometry
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from torch.autograd import Variable
import numpy as np

from math import exp
from skimage.metrics import structural_similarity
from skimage.measure.simple_metrics import peak_signal_noise_ratio
from guided_filter_pytorch.guided_filter import GuidedFilter


# Definition of Frequency Reconstruction Loss
class Freq_Recon_loss(nn.Module):
    def __init__(self):
        super(Freq_Recon_loss, self).__init__()

    def forward(self, input, target):
        input_fft = fft_spectrum(input)
        target_fft = fft_spectrum(target)

        loss = torch.abs(torch.subtract(target_fft, input_fft))
        loss = torch.log(loss + 1)

        min, max = float(loss.min()), float(loss.max())
        loss.clamp_(min = min, max=max)
        loss.add_(-min).div_(max-min+1e-5)

        return torch.mean(loss)


# Definition of Perceptual Loss
class VGG_loss(torch.nn.Module):
    def __init__(self, vgg):
        super(VGG_loss, self).__init__()
        self.vgg = vgg
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
    
    def forward(self, input, target):
        img_vgg = vgg_preprocess(input)
        target_vgg = vgg_preprocess(target)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


# Definition of SSIM Loss
class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# Definition of Total Variation Loss
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


# Guided filter
class GF(nn.Module):
    # https://pypi.org/project/guided-filter-pytorch/
    def __init__(self, r:int = 5, eps:float=2e-1):
        super(GF, self).__init__()
        self.g = GuidedFilter(r, eps)
    
    def forward(self, x, y):
        return self.g(x, y)


# Random Color Shift Algorithm before feeding input to texture discriminator
class ColorShift(nn.Module):
    def __init__(self, device):
        super(ColorShift, self).__init__()
        self.device = device
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        r1, g1, b1 = input[:,0,:,:].unsqueeze(1), input[:,1,:,:].unsqueeze(1), input[:,2,:,:].unsqueeze(1)
        r2, g2, b2 = target[:,0,:,:].unsqueeze(1), target[:,1,:,:].unsqueeze(1), target[:,2,:,:].unsqueeze(1)
        
        # uniform random values
        b_weight = torch.FloatTensor(1).uniform_(0.014, 0.214).to(self.device)
        r_weight = torch.FloatTensor(1).uniform_(0.199, 0.399).to(self.device)
        g_weight = torch.FloatTensor(1).uniform_(0.487, 0.687).to(self.device)

        output1 = (b_weight*b1+g_weight*g1+r_weight*r1)/(b_weight+g_weight+r_weight)
        output2 = (b_weight*b2+g_weight*g2+r_weight*r2)/(b_weight+g_weight+r_weight)
        
        return output1, output2



# Learning rate scheduling 
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )

# Fourier transform operation
def fft_spectrum(torch_img):

    out = fft.fftn(torch_img, dim=(2,3))
    _, _, h, w = out.shape
    center_h, center_w = h // 2, w // 2
    centered_out = torch.zeros_like(out)

    # shift spectrum
    centered_out[:, :, :center_h, :center_w] = out[:, :, center_h:, center_w:]
    centered_out[:, :, :center_h, center_w:] = out[:, :, center_h:, :center_w]
    centered_out[:, :, center_h:, :center_w] = out[:, :, :center_h, center_w:]
    centered_out[:, :, center_h:, center_w:] = out[:, :, :center_h, :center_w]

    return centered_out


# measure performance (PSNR and SSIM)
def calc_ssim(im1, im2):
    im1 = im1.data.cpu().detach().numpy().transpose(1, 2, 0)
    im2 = im2.data.cpu().detach().numpy().transpose(1, 2, 0)

    (score, _) = structural_similarity(im1, im2, full=True, multichannel=True)
    return score


def calc_psnr(im1, im2):
    im1 = im1.data.cpu().detach().numpy()
    im1 = im1[0].transpose(1, 2, 0)
    im2 = im2.data.cpu().detach().numpy()
    im2 = im2[0].transpose(1, 2, 0)
    return peak_signal_noise_ratio(im1, im2)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = batch * 255       #   * 0.5  [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))
    return batch


def calc_Freq(torch_img, kernel=3):
    if kernel == 3:
        sigma = 3
    elif kernel == 5:
        sigma = 1.5
    else:
        sigma = 1
    lowFreq = torchgeometry.image.gaussian_blur(
        torch_img, (kernel, kernel), (sigma, sigma)
    )
    highFreq = torch_img - lowFreq
    highFreq = RGB2gray(highFreq)
    return lowFreq, highFreq


# below codes are from SSD-GAN
# https://github.com/cyq373/SSD-GAN
def RGB2gray(rgb):
    if rgb.size(1) == 3:
        r, g, b = rgb[:,0,:,:], rgb[:,1,:,:], rgb[:,2,:,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    elif rgb.size(1) == 1:
        return rgb[:,0,:,:]


# Azimutal Averaging Operation
def azimuthalAverage(image, center=None):
    # Calculate the indices from the image
    H, W = image.shape[0], image.shape[1]
    y, x = np.indices([H, W])
    radius = np.sqrt((x - H/2)**2 + (y - W/2)**2)
    radius = radius.astype(np.int).ravel()
    nr = np.bincount(radius)
    tbin = np.bincount(radius, image.ravel())
    radial_prof = tbin / (nr + 1e-10)
    return radial_prof[1:-2]


def get_fft_feature(x):
    x_rgb = x.detach()
    epsilon = 1e-8

    x_gray = RGB2gray(x_rgb)
    fft = torch.rfft(x_gray,2,onesided=False)
    fft += epsilon
    magnitude_spectrum = torch.log((torch.sqrt(fft[:,:,:,0]**2 + fft[:,:,:,1]**2 + 1e-10))+1e-10)
    magnitude_spectrum = shift(magnitude_spectrum)
    magnitude_spectrum = magnitude_spectrum.cpu().numpy()

    out = []
    for i in range(magnitude_spectrum.shape[0]):
        out.append(torch.from_numpy(azimuthalAverage(magnitude_spectrum[i])).float().unsqueeze(0))
    out = torch.cat(out, dim=0)
    
    out = (out - torch.min(out, dim=1, keepdim=True)[0]) / (torch.max(out, dim=1, keepdim=True)[0] - torch.min(out, dim=1, keepdim=True)[0])
    out = Variable(out, requires_grad=True).to(x.device)
    
    return out


def shift(x: torch.Tensor):
    out = torch.zeros_like(x)

    H, W = x.size(-2), x.size(-1)
    out[:,:int(H/2),:int(W/2)] = x[:,int(H/2):,int(W/2):]
    out[:,:int(H/2),int(W/2):] = x[:,int(H/2):,:int(W/2)]
    out[:,int(H/2):,:int(W/2)] = x[:,:int(H/2),int(W/2):]
    out[:,int(H/2):,int(W/2):] = x[:,:int(H/2),:int(W/2)]
    return out