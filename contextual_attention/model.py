# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# local imports
import contextual_attention.model_layers as m_layers
from timeit import default_timer as timer

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
        # print("one: " + str(ones.shape))
        # print("MASK: " + str(mask.shape))
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
    def __init__(self, in_ch = 5, first_ch=32, out_ch=3, ca_weights = None):
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

        if(ca_weights):
        	self.load_ca_weights(ca_weights)

    def forward(self, x, mask=None):
        """
        # Inputs: 
        # - x: output image from the coarse net
        # - mask: resized mask from the coarse net
        """
        # forward to the dilation branch
        dil_out = self.dilation_encoder(x)

        # forward to the contextual attention branch
        attn_out = self.attention_encoder(x)

        attn_out, attn_flow = self.contextual_attention(
            attn_out, attn_out, mask=mask, rate=2)

        # concatenate both outputs on the channel dimension then feed to decoder
        refine_out = torch.cat([dil_out, attn_out], dim=1)
        refine_out = self.refine_decoder(refine_out)
        return refine_out, attn_flow

    def build_dilation_encoder(self, in_ch):
        layers = []

        down_out, curr_ch = m_layers.make_downsample_layers(in_ch)
        layers.append(down_out)

        layers.append(m_layers.make_atrous_layers(curr_ch))

        return nn.Sequential(*layers)

    def load_ca_weights(self, ca_weights):
        print("Loading Contextual Attention Module Weights. . .")
        is_cuda = False
        if torch.cuda.is_available():
            is_cuda = True
        # Dilation Branch (xconv in tf model)
        key = 'xconv'
        i = 1
        for module in self.dilation_encoder:
            # iterate through the sequentials
            for seq_module in module.children():
                if isinstance(seq_module, nn.Conv2d):
                    key_string = key + str(i)
                    if(i == 2 or i == 4):
                        key_string = key_string + "_downsample"
                    if(i <= 10 and i >= 7):
                        key_string = key_string + "_atrous"

                    seq_module.weight.data = torch.FloatTensor(ca_weights[key_string + "/kernel:0"])
                    if(is_cuda):
                        seq_module.weight.data = seq_module.weight.data.cuda()
                    seq_module.bias.data = torch.FloatTensor(ca_weights[key_string + "/bias:0"])
                    i += 1

        # Attention Branch
        key = 'pmconv'
        i = 1
        # Attention encoder
        for module in self.attention_encoder:
            # iterate through the sequentials
            if isinstance(module, nn.Conv2d):
                key_string = key + str(i)
                if(i == 2 or i == 4):
                    key_string = key_string + "_downsample"

                module.weight.data = torch.FloatTensor(ca_weights[key_string + "/kernel:0"])
                module.bias.data = torch.FloatTensor(ca_weights[key_string + "/bias:0"])
                i += 1
                if(is_cuda):
                    module.weight.data = module.weight.data.cuda()
                    
        # Contextual Attention Module
        i = 9
        for module in self.contextual_attention.final_layers:
            if isinstance(module, nn.Conv2d):
                key_string = key + str(i)
                module.weight.data = torch.FloatTensor(ca_weights[key_string + "/kernel:0"])
                module.bias.data = torch.FloatTensor(ca_weights[key_string + "/bias:0"])
                i += 1
                print("key: {}".format(key_string))
                if(is_cuda):
                    module.weight.data = module.weight.data.cuda()
                    
        # Upsampling
        key = "allconv"
        for module in self.refine_decoder:
            if isinstance(module, nn.Conv2d):
                key_string = key + str(i)
                if(i == 13 or i == 15):
                    key_string = "{}_upsample/{}_upsample_conv".format(key_string, key_string)
                module.weight.data = torch.FloatTensor(ca_weights[key_string + "/kernel:0"])
                module.bias.data = torch.FloatTensor(ca_weights[key_string + "/bias:0"])
                print("key: {}".format(key_string))
                i += 1
                if(is_cuda):
                    module.weight.data = module.weight.data.cuda()