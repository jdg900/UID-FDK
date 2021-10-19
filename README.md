# Unsupervised-Image-Denoising-With-Frequency-Domain-Knowledge (BMVC 2021 Oral)
This repository provides the official PyTorch implementation of the following paper:
>**Unsupervised Image Denoising With Frequency Domain Knowledge**
>
>Nahyun Kim* (KAIST), Donggon Jang* (KAIST), Sunhyeok Lee (KAIST), Bomi Kim (KAIST), and Dae-Shik Kim (KAIST) (*The authors have equally contributed.)
>
>BMVC 2021, Accepted as Oral Paper.

>**Abstract:** Supervised learning-based methods yield robust denoising results, yet they are inherently limited by the need for large-scale clean/noisy paired datasets. The use of unsupervised denoisers, on the other hand, necessitates a more detailed understanding of the underlying image statistics. In particular, it is well known that apparent differences between clean and noisy images are most prominent on high-frequency bands, justifying the use of low-pass filters as part of conventional image preprocessing steps. However, most learning-based denoising methods utilize only one-sided information from the spatial domain without considering frequency domain information. To address this limitation, in this study we propose a frequency-sensitive unsupervised denoising method. To this end,  a generative adversarial network (GAN) is used as a base structure. Subsequently, we include spectral discriminator and frequency reconstruction loss to transfer frequency knowledge into the generator. Results using natural and synthetic datasets indicate that our unsupervised learning method augmented with frequency information achieves state-of-the-art denoising performance, suggesting that frequency domain information could be a viable factor in improving the overall performance of unsupervised learning-based methods.

<p align="center">
    <img src="./figure/figure1.PNG">
</p>

## Requirements
To install requirements:

```setup
conda env create -n [your env name] -f environment.yaml
conda activate [your env name]
```

## To evaluate the model
### Synthetic Noise (AWGN)
Run this command:
```
sh test_awgn_sigma15.sh # AWGN with a noise level = 15
sh test_awgn_sigma25.sh # AWGN with a noise level = 25
sh test_awgn_sigma50.sh # AWGN with a noise level = 50
```

### Real-World Noise
Download the SIDD test dataset for evaluation in [here](https://drive.google.com/drive/folders/1lNet_6YH-sAG3nkR1zb2EKSiFmek7ywQ?usp=sharing) and place the dataset in `./dataset/test` directory.
After that, run this command:
```
sh test_real.sh
```

## Pre-trained model
We provide pre-trained models in `./checkpoints` directory.
```
checkpoints
|   AWGN_sigma15.pth # pre-trained model (AWGN with a noise level = 15)
|   AWGN_sigma25.pth # pre-trained model (AWGN with a noise level = 25)
|   AWGN_sigma50.pth # pre-trained model (AWGN with a noise level = 50)
|   SIDD.pth # pre-trained model (Real-World noise)
```

## Acknowledgements
This code is built on [U-GAT-IT](https://github.com/znxlwm/UGATIT-pytorch),[CARN](https://github.com/nmhkahn/CARN-pytorch), [SSD-GAN](https://github.com/cyq373/SSD-GAN). We thank the authors for sharing their codes.

We appreciate providing datasets [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [CBSD68](https://github.com/clausmichele/CBSD68-dataset), and [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/).

## Contact
If you have any questions, feel free to contact me (jdg900@kaist.ac.kr)