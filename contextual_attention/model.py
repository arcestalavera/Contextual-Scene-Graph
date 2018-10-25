# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# local imports
import contextual_attention.model_layers as m_layers


class Generator(nn.Module):
    def __init__(self, in_ch=5, first_ch=32):
        super(Generator, self).__init__()

        # stage 1: Coarse Network
        self.coarse_net = CoarseNetwork(in_ch, first_ch=first_ch)
        # print(self.coarse_net)

        # stage 2: Refinement network
        self.refinement_net = RefineNetwork(in_ch).cuda()

    def forward(self, x, mask):
        # Inputs for the network
        img_shape = x.shape
        ones = Variable(torch.ones(
            (img_shape[0], 1, img_shape[2], img_shape[3])).cuda())
        print("one: " + str(ones.shape))
        print("MASK: " + str(mask.shape))
        # concatenate input and mask
        net_in = torch.cat([x, ones, ones * mask], dim=1)

        # input to stage 1 (coarse network)
        coarse_out, coarse_mask = self.coarse_net(net_in, mask)

        print("coarse out = " + str(coarse_out.shape))

        # input to stage 2 (refinement network)
        refine_in = coarse_out * mask + x * (1. - mask)
        refine_in = torch.cat([refine_in, ones, ones * mask], dim=1)
        refine_out, _ = self.refinement_net(refine_in, coarse_mask)

        print("refine: " + str(refine_out.shape))
        return coarse_out, refine_out

class CoarseNetwork(nn.Module):
    def __init__(self, in_ch=5, first_ch=32, out_ch=3):
        super(CoarseNetwork, self).__init__()

        self.down_layers, curr_ch = m_layers.make_downsample_layers(
            in_ch, first_ch=first_ch)

        # final layers = atrous + upsample layers
        self.final_layers = []
        self.final_layers.append(m_layers.make_atrous_layers(curr_ch))
        self.final_layers.append(
            m_layers.make_upsample_layers(curr_ch, first_ch, out_ch))
        self.final_layers = nn.Sequential(*self.final_layers)

    def forward(self, x, mask):
        coarse_out = self.down_layers(x)
        print("COURSE OUT = " + str(coarse_out.shape))

        coarse_mask = F.interpolate(
            mask, size=[coarse_out.shape[2], coarse_out.shape[3]], mode='nearest')
        coarse_out = torch.clamp(self.final_layers(coarse_out), min=-1, max=1)

        return coarse_out, coarse_mask


class RefineNetwork(nn.Module):
    def __init__(self, in_ch = 5, first_ch=32, out_ch=3):
        super(RefineNetwork, self).__init__()

        # Dilation Branch
        self.dilation_encoder = self.build_dilation_encoder(in_ch)

        # Contextual Attention Branch
        self.attention_encoder, curr_ch = m_layers.make_downsample_layers(
            in_ch, activation=nn.ReLU())
        self.contextual_attention = m_layers.AttentionModule(curr_ch)

        # Decoder of Refinement Network
        self.refine_decoder = m_layers.make_upsample_layers(
            curr_ch * 2, first_ch * 2, out_ch) #change back to curr * 2, first * 2

    def forward(self, x, mask=None):
        """
        # Inputs: 
        # - x: output image from the coarse net
        # - mask: resized mask from the coarse net
        """
        # forward to the dilation branch
        dil_out = self.dilation_encoder(x)

        # print("DIL BRANCH: " + str(dil_out.shape))
        # forward to the contextual attention branch
        attn_out = self.attention_encoder(x)
        attn_out, attn_flow = self.contextual_attention(
            attn_out, attn_out, mask=mask, rate=2)

        # print("ATTENTION BRANCH: " + str(attn_out.shape))
        # concatenate both outputs on the channel dimension then feed to decoder
        refine_out = torch.cat([dil_out, attn_out], dim=1)
        refine_out = self.refine_decoder(refine_out)

        # print("REFINE OUT: " + str(refine_out.shape))

        return refine_out, attn_flow

    def build_dilation_encoder(self, in_ch):
        layers = []

        down_out, curr_ch = m_layers.make_downsample_layers(in_ch)
        layers.append(down_out)

        layers.append(m_layers.make_atrous_layers(curr_ch))

        return nn.Sequential(*layers)
