import argparse
import torch
import random
import os
from UDG import UDG


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

    """ test setting """
    parser.add_argument('--exp', required=True, help='experiment name' )
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of workers')
    parser.add_argument('--datarootN_val', type=str, default='', help='path for validation folder')
    parser.add_argument('--sigma', type=int, default=50, help='noise level')
    parser.add_argument('--real', type=str2bool, default=False, help='real(SIDD) image training')
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
    
    os.makedirs(args.testimagepath, exist_ok=True)
    

def main():
    args = parse_args()

    prepare(args)

    udg = UDG(args)

    udg.build_model(args)

    if args.test:
        print("Test Started")
        udg.test(args)


if __name__=="__main__":
    main()