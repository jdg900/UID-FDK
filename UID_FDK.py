import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from models import *
from utils import *
from dataloader import *

import time
import itertools
import sys
import datetime
import os


class UID_FDK(object):
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
            
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(args.patch_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            valid_transform = transforms.Compose([transforms.ToTensor()])

            if args.real:
                self.train_dataset = realDataset_from_h5(
                    args.datarootN,
                    args.datarootC,
                    transform=train_transform,
                )

                self.val_dataset = real_Dataset_test(
                    args.datarootN_val,  
                    transform=valid_transform,
                )

            else:
                self.train_dataset = Dataset_from_h5(
                    args.datarootN,
                    args.datarootC,
                    sigma=args.sigma,
                    transform=train_transform,
                )

                self.val_dataset = Dataset_test(
                    args.datarootN_val, sigma=args.sigma, transform=valid_transform
                )

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_cpu,
            )

            self.val_loader = DataLoader(
                self.val_dataset, batch_size=1, shuffle=False, num_workers=args.n_cpu
            )

        """ Define Generator & Discriminator """
        self.genN2C = Generator_N2C()

        self.genC2N = Generator_C2N(
            input_channel=3, 
            output_channel=3, 
            middle_channel=args.ch_genn2c, 
            n_blocks=args.n_res_gen
        )

        self.disC = Discriminator(
            input_nc=3, ndf=args.ch_dis, n_layers=args.n_conv_dis
        )
        self.disT = Discriminator(
            input_nc=1, ndf=args.ch_dis, n_layers=args.n_conv_dis
        )
        self.disS = Spectral_Discriminator(args.patch_size)

        self.genN2C = nn.DataParallel(self.genN2C)
        self.genC2N = nn.DataParallel(self.genC2N)

        self.disC = nn.DataParallel(self.disC)
        self.disT = nn.DataParallel(self.disT)
        self.disS = nn.DataParallel(self.disS)

        """ Device """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        """ Define Loss """
        self.MSE_Loss = nn.MSELoss().to(self.device)  # LSGAN loss
        self.L1_Loss = nn.L1Loss().to(self.device)
        self.TV_Loss = TVLoss().to(self.device)
        self.SSIM_Loss = SSIM_loss().to(self.device)

        vgg = vgg_19()
        if torch.cuda.is_available():
            vgg.to(self.device)

        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        self.VGG_Loss = VGG_loss(vgg).to(self.device)
        self.ColorShift = ColorShift(device=self.device)
        self.background = GF(r=10, eps=0.3)
        self.Freq_Recon_loss = Freq_Recon_loss()

        """ Optimizer """
        self.G_optim = optim.Adam(
            itertools.chain(
            self.genN2C.parameters(), self.genC2N.parameters()
            ),
            lr=args.lrG,
        )
        self.D_optim = optim.Adam(
            itertools.chain(
                self.disC.parameters(), self.disT.parameters(), self.disS.parameters()
            ),
            lr=args.lrD,
            betas=(0.5, 0.999),
            weight_decay=0.0001,
        )

    def train(self, args):
        self.genN2C.train().to(self.device), self.genC2N.train().to(self.device)
        self.disC.train().to(self.device), self.disT.train().to(self.device), self.disS.train().to(self.device)

        """ CheckPoint """
        ckpt_path = os.path.join(args.modelsavepath, "last.pth")
        last_epoch = 0
        best_epoch = 0
        best_PSNR = 0.0
        best_SSIM = 0.0

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            self.genN2C.load_state_dict(ckpt["genN2C"])
            self.genC2N.load_state_dict(ckpt["genC2N"])
            self.disC.load_state_dict(ckpt["disL"])
            self.disT.load_state_dict(ckpt["disH"])            
            self.disS.load_state_dict(ckpt["disFFT"])
            self.G_optim.load_state_dict(ckpt["G_optimizer"])
            self.D_optim.load_state_dict(ckpt["D_optimizer"])
            last_epoch = ckpt["epoch"] + 1
            best_PSNR = ckpt["best_PSNR"]
            best_SSIM = ckpt["best_SSIM"]
            args.seed = ckpt["seed"]
            print("Last checkpoint is loaded. start_epoch:", last_epoch)

        else:
            print("No checkpoint is found.")

        milestones = [epoch - last_epoch for epoch in eval(args.decay_step)]
        decay_epoch = milestones[0]
        G_schedular = torch.optim.lr_scheduler.LambdaLR(
            self.G_optim, lr_lambda=LambdaLR(args.n_epochs, 0, decay_epoch).step
        )
        D_schedular = torch.optim.lr_scheduler.LambdaLR(
            self.D_optim, lr_lambda=LambdaLR(args.n_epochs, 0, decay_epoch).step
        )

        """ Training Loop """

        prev_time = time.time()
        
        for epoch in range(last_epoch, args.n_epochs):
            self.genN2C.train(), self.genC2N.train()
            self.disC.train(), self.disT.train(), self.disS.train()
     
            for G_param_group, D_param_group in zip(
                self.G_optim.param_groups, self.D_optim.param_groups
            ):
                print(
                    " ***SEED: %d  LRG: %f  LRD : %f***"
                    % (args.seed, G_param_group["lr"], D_param_group["lr"])
                )
                f = open("%s/log.txt" % args.logpath, "at", encoding="utf-8")
                f.write(
                    "\n ***SEED: %d  LRG: %f  LRD : %f***"
                    % (args.seed, G_param_group["lr"], D_param_group["lr"])
                )

            for i, (Noisy, Clean) in enumerate(self.train_loader):
                
                Noisy = Noisy.to(self.device)
                Clean = Clean.to(self.device)

                # Update D
                self.D_optim.zero_grad(set_to_none=True)

                fake_N2C = self.genN2C(Noisy)

                real_img_logit = self.disC(Clean)
                fake_img_logit = self.disC(fake_N2C.detach())
                
                clean_texture, fake_n2c_texture = self.ColorShift(Clean, fake_N2C)
                real_texture_logit = self.disT(clean_texture)
                fake_texture_logit = self.disT(fake_n2c_texture.detach())
                
                real_spectral_logit = self.disS(Clean)
                fake_spectral_logit = self.disS(fake_N2C.detach())
                
                del clean_texture
                
                D_ad_loss_img = self.MSE_Loss(
                    real_img_logit, torch.ones_like(real_img_logit)
                ) + self.MSE_Loss(
                    fake_img_logit, torch.zeros_like(fake_img_logit)
                )
                D_ad_loss_texture = self.MSE_Loss(
                    real_texture_logit, torch.ones_like(real_texture_logit)
                ) + self.MSE_Loss(
                    fake_texture_logit, torch.zeros_like(fake_texture_logit)
                )
                D_ad_loss_spectral = self.MSE_Loss(
                    real_spectral_logit, torch.ones_like(real_spectral_logit)
                ) + self.MSE_Loss(
                    fake_spectral_logit, torch.zeros_like(fake_spectral_logit)
                )

                (args.adv_w * D_ad_loss_img).backward()
                (args.fft_w * D_ad_loss_spectral).backward()
                (args.texture_w * D_ad_loss_texture).backward()
                    
                self.D_optim.step()

                # Update G
                self.G_optim.zero_grad(set_to_none=True)
                
                G_BackGround_loss = self.L1_Loss(self.background(Noisy, Noisy), self.background(fake_N2C, fake_N2C))

                fake_img_logit = self.disC(fake_N2C)
                fake_texture_logit = self.disT(fake_n2c_texture)
                fake_spectral_logit = self.disS(fake_N2C)

                fake_N2C2N = self.genC2N(fake_N2C)
                fake_N2C_low, _ = calc_Freq(fake_N2C, 5)
                noisy_low, _ = calc_Freq(Noisy, 5)
                G_BackGround_loss += self.L1_Loss(noisy_low, fake_N2C_low)

                del fake_n2c_texture
                
                G_ad_loss_img = self.MSE_Loss(fake_img_logit, torch.ones_like(fake_img_logit))
                G_ad_loss_texture = self.MSE_Loss(
                    fake_texture_logit, torch.ones_like(fake_texture_logit)
                )
                
                G_ad_loss_spectral = self.MSE_Loss(
                    fake_spectral_logit, torch.ones_like(fake_spectral_logit)
                )
                G_ad_loss = args.adv_w * G_ad_loss_img + args.fft_w * G_ad_loss_spectral + args.texture_w * G_ad_loss_texture
                        
                G_cycle_loss = self.L1_Loss(fake_N2C2N, Noisy)
                G_freq_loss = self.Freq_Recon_loss(fake_N2C2N, Noisy)
                G_vgg_loss = self.VGG_Loss(Noisy, fake_N2C)
                G_TV_loss = self.TV_Loss(fake_N2C)
                G_ssim_loss = 1 - self.SSIM_Loss(fake_N2C2N, Noisy)

                del fake_N2C
                del fake_N2C2N
                
                Generator_loss = (
                    G_ad_loss
                    + args.vgg_w * G_vgg_loss
                    + args.back_w * G_BackGround_loss
                    + args.cycle_w * G_cycle_loss
                    + args.freq_w * G_freq_loss
                    + args.tv_w * G_TV_loss
                    + args.ssim_w * G_ssim_loss
                )

                Generator_loss.backward()
                self.G_optim.step()

                """ evaluation & checkpoint save & image write """
                batches_done = epoch * len(self.train_loader) + i
                batches_left = args.n_epochs * len(self.train_loader) - batches_done
                time_left = datetime.timedelta(
                    seconds=(batches_left) * (time.time() - prev_time)
                )
                prev_time = time.time()
                
                sys.stdout.write(
                    "\r[Epoch: {}/{}] [Batch: {}/{}] [ETA: {}]".format(
                        epoch,
                        args.n_epochs,
                        i,
                        len(self.train_loader),
                        time_left,
                    )
                )

            val_psnr, val_ssim = self.test(args)

            G_schedular.step()
            D_schedular.step()

            ckpt = {
                "genN2C": self.genN2C.state_dict(),
                "genC2N": self.genC2N.state_dict(),
                "disL": self.disC.state_dict(),
                "disH": self.disT.state_dict(),
                "disFFT": self.disS.state_dict(),
                "G_optimizer": self.G_optim.state_dict(),
                "D_optimizer": self.D_optim.state_dict(),
                "epoch": epoch,
                "best_PSNR": best_PSNR,
                "best_SSIM": best_SSIM,
                "seed": args.seed,
            }

            torch.save(ckpt, ckpt_path)

            if best_PSNR < val_psnr:
                best_PSNR = val_psnr
                best_SSIM = val_ssim
                best_epoch = epoch
                torch.save(
                    self.genN2C.state_dict(),
                    "{}/best_G_N2C.pth".format(args.modelsavepath),
                )
                
            print(
                "\n=== Best PSNR: %.4f, SSIM: %.4f at epoch %d === "
                % (best_PSNR, best_SSIM, best_epoch)
            )
            f = open("%s/log.txt" % args.logpath, "at", encoding="utf-8")
            f.write(
                "\n === Best PSNR: %.4f, SSIM: %.4f at epoch %d ==="
                % (best_PSNR, best_SSIM, best_epoch)
            )

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