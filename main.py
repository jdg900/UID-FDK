import argparse
import torch
import random
import os

from utils import *
from UID_FDK import UID_FDK


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()

    """ environment setting"""
    parser.add_argument('--exp', required=True, help='experiment name' )
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of workers')
    parser.add_argument('--patch_size', type=int, default=128, help='patch size')
    parser.add_argument('--datarootC', type=str, default='', help='path for clean images folder')
    parser.add_argument('--datarootN', type=str, default='', help='path for noisy images foloder')
    parser.add_argument('--datarootN_val', type=str, default='', help='path for validation folder')

    """ train setting """
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--sigma', type=int, default=50, help='noise level')
    parser.add_argument('--real', type=str2bool, default=False, help='real(SIDD) image training')

    parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate of generator')
    parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate of discriminator')
    parser.add_argument('--decay_step', default="[70]", help='learning rate decay step')

    parser.add_argument('--n_res_gen', type=int, default=2, help='number of resblock in C2N generator')
    parser.add_argument('--n_conv_dis', type=int, default=3, help='number of layers in N2C discriminator')
    parser.add_argument('--ch_genn2c', type=int, default=32, help='channel size in N2C generator')
    parser.add_argument('--ch_dis', type=int, default=64, help='channel size in discriminator')
    
    parser.add_argument('--adv_w', type=float, default=1.0, help='weight of clean adversarial loss')
    parser.add_argument('--texture_w', type=float, default=1.0, help='weight of texture adversarial loss')
    parser.add_argument('--fft_w', type=float, default=1.0, help='weight of spectral adversarial loss')
    parser.add_argument('--vgg_w', type=float, default=2.0, help='weight of vgg19 loss')
    parser.add_argument('--tv_w', type=float, default=0.2, help='weight of total variance loss')
    parser.add_argument('--back_w', type=float, default=2.0, help='weight of background preserving loss')
    parser.add_argument('--cycle_w', type=float, default=1.0, help='weight of cycle loss')
    parser.add_argument('--ssim_w', type=float, default=0.2, help='weight of ssim loss')
    parser.add_argument('--freq_w', type=float, default=0.2, help='weight of freqeuncy recon loss')
    

    """ test setting """
    parser.add_argument('--test', type=str2bool, default=False, help='test mode')
    parser.add_argument('--testmodel_n2c', default=0, help='path for n2c test model')

    args = parser.parse_args()

    return args


def prepare(args):
    if args.seed == 0:
        args.seed = random.randint(1, 10000)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.logpath = os.path.join("./exp", args.exp, str(args.seed))
    args.testimagepath = os.path.join(args.logpath, "testimages")
    args.modelsavepath = os.path.join(args.logpath, "saved_models")
    
    os.makedirs(args.testimagepath, exist_ok=True)
    os.makedirs(args.modelsavepath, exist_ok=True)


def main():
    args = parse_args()

    prepare(args)

    udg = UID_FDK(args)

    udg.build_model(args)

    if args.test:
        print("Test Started")
        udg.test(args)

    else:
        print("Train Started")
        udg.train(args)


if __name__=="__main__":
    main()