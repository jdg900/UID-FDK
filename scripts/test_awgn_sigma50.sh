CUDA_VISIBLE_DEVICES=0 python main.py --exp pre_trained_awgn_sigma50 --datarootN_val ./dataset/test/CBSD68 --sigma 50 --test 1 --testmodel_n2c ./checkpoints/AWGN_sigma50.pth
