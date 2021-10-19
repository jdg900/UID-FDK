import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from models import *
from utils import *
from dataloader import *


class UDG(object):
    def __init__(self, args):
        print(args)

    def build_model(self, args):
        """ Transform & Dataloader """
        if args.test:
            test_transform = transforms.Compose([transforms.ToTensor()])
            if args.real:
                self.test_dataset = real_Dataset_test(
                    args.datarootN_val, transform=test_transform
                )
            else:
                self.test_dataset = Dataset_test(
                    args.datarootN_val, sigma=args.sigma, transform=test_transform
                )

            self.val_loader = DataLoader(
                self.test_dataset, batch_size=1, shuffle=False, num_workers=args.n_cpu
            )
            
        """ Define Generator N2C"""
        self.genN2C = Generator_N2C()
        self.genN2C = nn.DataParallel(self.genN2C)

        """ Device """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

    def test(self, args):
        if args.test:
            print("test model loading")
            testmodel_n2c = args.testmodel_n2c
            self.genN2C.load_state_dict(torch.load(testmodel_n2c))

        self.genN2C.eval().to(self.device)

        cumulative_psnr = 0
        cumulative_ssim = 0

        with torch.no_grad():
            for i, (Clean, Noisy) in enumerate(self.val_loader):
                _, _, h, w = Clean.size()
                if not args.real:
                    Clean = F.interpolate(Clean, (h - 1, w - 1)).to(self.device)
                    Noisy = F.interpolate(Noisy, (h - 1, w - 1)).to(self.device)

                Clean = Clean.to(self.device)
                Noisy = Noisy.to(self.device)

                output_n2c= self.genN2C(Noisy)
                output_n2c = torch.clamp(output_n2c, 0, 1)

                if args.test:
                    save_image(
                        output_n2c[0],
                        "{}/Bestn2c_{}.png".format(args.testimagepath, i),
                        nrow=1,
                        normalize=True,
                    )
                    
                cur_psnr = calc_psnr(output_n2c, Clean)
                cur_ssim = calc_ssim(output_n2c[0], Clean[0])

                cumulative_psnr += cur_psnr
                cumulative_ssim += cur_ssim

            val_psnr = cumulative_psnr / len(self.val_loader)
            val_ssim = cumulative_ssim / len(self.val_loader)

        print("PSNR : {}, SSIM : {}".format(val_psnr, val_ssim))

        return val_psnr, val_ssim