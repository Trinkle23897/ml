# Unsupervised learning

[TOC]

## Generate Model

### PixelRNN and PixelCNN

根据相邻像素确定当前像素分布概率

### VAE

训练encoder和decoder，输入中间自定义feature，输出图像

### GAN

$$
\displaystyle\min_{\theta_g}\max_{\theta_d}[\mathbb{E}_{x\sim p_{data}}\log{D_{\theta_d}(x)}+\mathbb{E}_{z\sim p(z)}\log(1-D_{\theta_d}(G_{\theta_g}(z)))]
$$

where:

- Discriminator outputs likelihood in (0,1) of real image
- $D_{\theta_d}(x)$: Discriminator output for real data x
- $D_{\theta_d}(G_{\theta_g}(z))$: Discriminator output for generated fake data G(z)
- Discriminator (θ d ) wants to maximize objective such that D(x) is close to 1 (real) and D(G(z)) is close to 0 (fake)
- Generator (θ g ) wants to minimize objective such that D(G(z)) is close to 1 (discriminator is fooled into thinking generated G(z) is real)

DCGAN: 加了CNN “Radford et al, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”, ICLR 2016”

https://github.com/hindupuravinash/the-gan-zoo

See also: https://github.com/soumith/ganhacks for tips and tricks for trainings GANs

## RL

