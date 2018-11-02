import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import contextual_attention.helper as helper
import utils

# import numpy as np

class AttentionModule(nn.Module):
    def __init__(self, in_ch):
        super(AttentionModule, self).__init__()
        self.final_layers = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1))

        self.zero_pad = nn.ZeroPad2d(1)

    def forward(self, f, b, mask=None, ksize=3, stride=1, rate=1,
                fuse_k=3, softmax_scale=10., training=True, fuse=True):

        """ Contextual attention layer implementation.

        Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

        Args:
        f: Input feature to match (foreground).
        b: Input feature for match (background).
        mask: Input mask for b, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

        Returns:
        tf.Tensor: output
        """

        # get shapes of foreground (f) and background (b)
        raw_fs = f.shape
        # print("RAW FS: " + str(raw_fs))
        raw_int_fs = list(f.shape)
        raw_int_bs = list(b.shape)

        # extract 3x3 patches from background with stride and rate
        kernel = 2 * rate
        raw_w = self.extract_image_patches(b, kernel, rate * stride)

        # Reshape raw_w to match pytorch conv weights shape
        raw_w = torch.reshape(
            raw_w, [raw_int_bs[0], -1, raw_int_bs[1], kernel, kernel])  # b x in_ch (h * w) x out_ch (c) x k x k

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1. / rate, mode='nearest')
        b = F.interpolate(
            b, size=[int(raw_int_bs[2] / rate), int(raw_int_bs[3] / rate)], mode='nearest')

        # get shape of foreground then split on the batch dimension
        fs = f.shape
        int_fs = list(f.shape)
        f_groups = torch.split(f, 1, dim=0)

        # print("F GROUPS: " + str(f_groups[0].shape))


        bs = b.shape
        int_bs = list(b.shape)

        # extract w then reshape to weight shape of functional conv2d of pytorch
        w = self.extract_image_patches(b, ksize, stride)
        # reshape to b x in_ch (h * w) x out_ch (c) x k x k
        # print("INT FS: " + str(int_fs))
        w = torch.reshape(w, [int_fs[0], -1, int_fs[1], ksize, ksize])

        # print("W: " + str(w.shape))
        # process mask
        if mask is None:
            mask = torch.zeros([bs[0], 1, bs[2], bs[3]]).cuda()
        else:
            # print("DOWNSAMPLE MEN")
            mask = F.interpolate(mask, scale_factor = 1. / rate, mode = 'nearest')

        m = self.extract_image_patches(mask, ksize, stride)

        # make mask have the shape of (b x c x hw x k x k)
        # print("m = " + str(mask.shape))
        if(mask.shape[0] > 1):
            m = torch.reshape(m, [mask.shape[0], 1, -1, ksize, ksize])
        else:
            m = torch.reshape(m, [1, 1, -1, ksize, ksize])
        # m = m[0]
        # print("MY M: " + str(m.shape))
        # create batch for mm
        mm = []
        for i in range(m.shape[0]):
            mm.append(utils.reduce_mean(m[i], axis = [0, 2, 3], keep_dims = True))

        mm = torch.cat(mm)

        # print("mm: " + str(mm.shape))
        w_groups = torch.split(w, 1, dim=0)
        raw_w_groups = torch.split(raw_w, 1, dim=0)
        y = []
        offsets = []
        k = fuse_k
        scale = softmax_scale
        fuse_weight = utils.to_var(torch.reshape(torch.eye(k), [1, 1, k, k]))

        for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mm):
            """
            # Conv per batch
            # VARIABLES:
            # - xi: input to the conv; tensors from foreground (f_groups)
            # - wi: weights for training; image patches from the background (w_groups): 
            # - raw_wi: patches from the background (raw_w_groups)
            """
            # conv for compare
            wi = wi[0] # 

            wi_normed = wi / \
                torch.max(torch.sqrt(utils.reduce_sum(
                    wi ** 2, axis=[0, 2, 3])), torch.FloatTensor([1e-4]).cuda())
            
            # print("wi_normed: " + str(wi_normed.shape))
            # print("xi:" + str(xi.shape))
            yi = F.conv2d(xi, wi_normed, stride=1, padding=1)
            # print("yi: " + str(yi.shape))
            # wi_normed = wi / torch.max(torch.sqrt(torch.sum(torch.square()))) #l2 norm
            # conv implementation for fuse scores to encourage large patches
            if fuse:
                # b x c x f(hw) x b(hw)
                yi = torch.reshape(yi, [1, 1, fs[2] * fs[3], bs[2] * bs[3]])
                # print("yi: " + str(yi.shape))
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)
                yi = torch.reshape(yi, [1, fs[2], fs[3], bs[2], bs[3]])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = torch.reshape(yi, [1, 1, fs[2] * fs[3], bs[2] * bs[3]])
                # print("yi: " + str(yi.shape))
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)
                yi = torch.reshape(yi, [1, fs[3], fs[2], bs[3], bs[2]])
                yi = yi.permute(0, 2, 1, 4, 3)
                # print("yi inside fuse: " + str(yi.shape))
                # print("yi: " + str(yi.shape))

            yi = torch.reshape(yi, [1, bs[2] * bs[3], fs[2], fs[3]])
            # print("yi: " + str(yi.shape))
            # softmax to match
            yi = yi * mi
            # print("hey")
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mi  # mask

            _, offset = torch.max(yi, dim=1)
            offset = torch.stack([offset // fs[3], offset % fs[3]], dim=-1)

            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=rate, padding=1) / 4.
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)

        offsets = torch.cat(offsets, dim=0)
        offsets = torch.reshape(
            offsets, [int_bs[0]] + [2] + int_bs[2:])  # skip channel

        # case1: visualize optical flow: minus current position
        # height
        h_add = utils.to_var(torch.reshape(torch.arange(bs[2]), [1, 1, bs[2], 1]))
        h_add = h_add.expand([bs[0], 1, bs[2], bs[3]])

        # width
        w_add = utils.to_var(torch.reshape(torch.arange(bs[3]), [1, 1, 1, bs[3]]))
        w_add = w_add.expand([bs[0], 1, bs[2], bs[3]])

        # concat on channel
        offsets = offsets - torch.cat([h_add, w_add], dim=1)
        
        # to flow image
        flow = helper.flow_to_image(offsets.permute(0, 2, 3, 1).data.cpu().numpy())
        flow = torch.from_numpy(flow).permute(0, 3, 1, 2)

        # case2: visualize which pixels are attended
        # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
        if rate != 1:
            flow = F.interpolate(flow, scale_factor=rate, mode='nearest')

        out = self.final_layers(y)
        return out, flow

    def extract_image_patches(self, image, kernel, stride):
        # pad image
        image = self.zero_pad(image)

        return image.unfold(2, kernel, stride).unfold(3, kernel, stride)

def make_downsample_layers(in_ch, out_ch=128, first_ch=32, activation=None):
    """
    # Make the downsampling layers (Conv layers 1 - 6 of the stages)
    # Variables:
    # - in_ch: number of input channels
    # - first_ch: number of channels on the first layer
    # - out_ch: number of output channels for this module
    # - activation: activation at the end of the module
    """
    layers = []
    curr_ch = first_ch  # current channels
    layers.append(nn.Conv2d(in_ch, curr_ch, kernel_size=5,
                            stride=1, padding=1))
    while(curr_ch < out_ch):
        layers.append(nn.Conv2d(curr_ch, curr_ch * 2,
                                kernel_size=3, stride=2, padding=1))
        layers.append(nn.Conv2d(curr_ch * 2, curr_ch * 2,
                                kernel_size=3, stride=1, padding=1))
        curr_ch *= 2

    layers.append(nn.Conv2d(curr_ch, curr_ch, kernel_size=3,
                            stride=1, padding=1))

    if activation != None:
        layers.append(activation)

    return nn.Sequential(*layers), curr_ch

def make_atrous_layers(in_ch, first_dil=2, final_dil=16):
    """
    # Make the atrous layers (Conv layers 7 - 10 of the stages)
    # Variables:
    # - in_ch: number of input channels
    # - first_dil: size of initial dilation
    # - final_dil: size of final dilation
    """
    curr_dil = first_dil

    layers = []
    while(curr_dil <= final_dil):
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size=3,
                                stride=1, padding=curr_dil, dilation=curr_dil))
        curr_dil *= 2

    return nn.Sequential(*layers)

def make_upsample_layers(in_ch, first_ch, out_ch=3):
    """
    # Make the atrous layers (Conv layers 11 - 17 of the stages)
    # Variables:
    # - in_ch: number of input channels
    # - first_ch: number of channels on the first conv layer
    # - out_ch: number of output channels on the output of this module
    """
    curr_ch = in_ch
    layers = []

    # Conv 11
    layers.append(nn.Conv2d(curr_ch, curr_ch, kernel_size=3,
                            stride=1, padding=1))

    # Conv 12 - 15
    while(curr_ch > first_ch):
        layers.append(nn.Conv2d(curr_ch, curr_ch,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(curr_ch, curr_ch // 2,
                                kernel_size=3, stride=1, padding=1))
        curr_ch = curr_ch // 2

    # Conv 16 - 17
    layers.append(nn.Conv2d(curr_ch, curr_ch // 2,
                            kernel_size=3, stride=1, padding=1))
    layers.append(nn.Conv2d(curr_ch // 2, out_ch,
                            kernel_size=3, stride=1, padding=1))

    return nn.Sequential(*layers)