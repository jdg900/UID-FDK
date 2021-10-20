import h5py
import matplotlib.image as mpimg
import os
import numpy as np
import glob


def crop_patch(img, img_size=(512, 512), patch_size=(150, 150), stride=150, random_crop=False):
    count = 0
    patch_list = []
    if random_crop == True:
        crop_num = 100
        pos = [(np.random.randint(patch_size, img_size[0] - patch_size), np.random.randint(patch_size, img_size[1] - patch_size))
               for i in range(crop_num)]
    else:
        pos = [(x, y) for x in range(patch_size[1], img_size[1] - patch_size[1], stride) for y in
               range(patch_size[0], img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt - patch_size[0]:yt + patch_size[0], xt - patch_size[1]:xt + patch_size[1]]
        patch_list.append(cropped_img)
        count += 1

    return patch_list


def gen_dataset(src_files, dst_path):
    h5py_name = dst_path
    h5f = h5py.File(h5py_name, 'w')

    for i in range(len(src_files)):
        print(src_files[i])
        img = mpimg.imread(src_files[i])
        [h, w, c] = img.shape
        patch_list = crop_patch(img, (h, w), (150, 150), 150, False)

        for num in range(len(patch_list)):
            data = patch_list[num].copy()
            h5f.create_dataset(str(i)+'_'+str(num), shape=(300,300,3), data=data)
    h5f.close()



if __name__ == "__main__":
    src_path = "./dataset/DIV2K_N/"
    dst_path = "./dataset/train/div2k_trainN.h5"

    os.makedirs(src_path, exist_ok=True)
    os.makedirs(dst_path, exist_ok=True)

    src_files = glob.glob(src_path + "*.PNG") + glob.glob(src_path + "*.png") 
    print("start noisy set...")
    gen_dataset(src_files, dst_path)
    print('end noisy set...')

    src_path = "./dataset/DIV2K_C/"
    dst_path = "./dataset/train/div2k_trainC.h5"

    os.makedirs(src_path, exist_ok=True)
    os.makedirs(dst_path, exist_ok=True)
    
    src_files = glob.glob(src_path + "*.PNG") + glob.glob(src_path + "*.png") 
    print("start clean set...")
    gen_dataset(src_files, dst_path)
    print('end clean set')