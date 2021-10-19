import torch.utils.data as data
import glob
import os
from PIL import Image
from natsort import natsorted


class Dataset_test(data.Dataset):
    def __init__(self, src_path, sigma, transform=None):
        self.src_path = natsorted(glob.glob(os.path.join(src_path, '*.png')))
        self.src_path_noise = natsorted(glob.glob(os.path.join(src_path, f'noisy_sig{sigma}', '*.png')))
        self.transform = transform

    def __getitem__(self, index):
        clean = Image.open(self.src_path[index])
        noisy = Image.open(self.src_path_noise[index])

        clean = self.transform(clean)
        noisy = self.transform(noisy)

        return (clean, noisy)
    
    def __len__(self):
        return len(self.src_path)


class real_Dataset_test(data.Dataset):
    def __init__(self, src_path, transform=None):
        self.src_path = natsorted(glob.glob(os.path.join(src_path, 'GT','*.png')))
        self.src_path_noise = natsorted(glob.glob(os.path.join(src_path, 'Noisy', '*.png')))
        self.transform = transform

    def __getitem__(self, index):
        clean = Image.open(self.src_path[index])
        noisy = Image.open(self.src_path_noise[index])

        clean = self.transform(clean)
        noisy = self.transform(noisy)

        return (clean, noisy)
    
    def __len__(self):
        return len(self.src_path)
