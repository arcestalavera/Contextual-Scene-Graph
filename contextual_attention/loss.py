import torch
import helper

def gan_wgan_loss(pos, neg):
    """
    wgan loss function for GANs.
    - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    """
    d_loss = torch.mean(pos - neg)
    g_loss = -torch.mean(neg)
    
    return g_loss, d_loss