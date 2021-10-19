from skimage.metrics import structural_similarity
from skimage.measure.simple_metrics import peak_signal_noise_ratio


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