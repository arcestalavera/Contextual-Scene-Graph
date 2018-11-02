# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Self
import scene_graph.model as SGModel
import contextual_attention.model as CAModel
from contextual_attention.critics import ContextualCritic
import utils

class ContextualSceneModel(nn.Module):

    def __init__(self, **kwargs):
        super(ContextualSceneModel, self).__init__()

        # scene graph model
        self.sg_model = SGModel.Sg2ImModel(vocab=kwargs['vocab'],
                                           image_size=kwargs['image_size'],
                                           embedding_dim=kwargs['embedding_dim'],
                                           gconv_dim=kwargs['gconv_dim'],
                                           gconv_hidden_dim=kwargs['gconv_hidden_dim'],
                                           gconv_num_layers=kwargs['gconv_num_layers'],
                                           mlp_normalization=kwargs['mlp_normalization'],
                                           refinement_network_dims=kwargs['refinement_network_dims'],
                                           normalization=kwargs['normalization'],
                                           activation=kwargs['activation'],
                                           mask_size=kwargs['mask_size'],
                                           layout_noise_dim=kwargs['layout_noise_dim'])

        # handle refinement network arguments
        self.ca_module = CAModel.RefineNetwork()
        # if(kwargs['ca_weights'] is not None):
        #     self.load_ca_weights(kwargs['ca_weights'])

        # self.foreground_objs = kwargs['foreground_objs']
    def load_ca_weights(self, ca_weights):
        print("Loading Contextual Attention Module Weights. . .")
        is_cuda = False
        if torch.cuda.is_available():
            is_cuda = True
        # Dilation Branch (xconv in tf model)
        key = 'xconv'
        i = 1
        for module in self.ca_module.dilation_encoder:
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
                    # seq_module.bias.data = torch.FloatTensor(ca_weights[key_string + "/bias:0"])
                    i += 1

        # Attention Branch
        key = 'pmconv'
        i = 1
        # Attention encoder
        for module in self.ca_module.attention_encoder:
            # iterate through the sequentials
            if isinstance(module, nn.Conv2d):
                key_string = key + str(i)
                if(i == 2 or i == 4):
                    key_string = key_string + "_downsample"

                module.weight.data = torch.FloatTensor(ca_weights[key_string + "/kernel:0"])
                # module.bias.data = torch.FloatTensor(ca_weights[key_string + "/bias:0"])
                i += 1
                if(is_cuda):
                    module.weight.data = module.weight.data.cuda()
                    
        # Contextual Attention Module
        i = 9
        for module in self.ca_module.contextual_attention.final_layers:
            if isinstance(module, nn.Conv2d):
                key_string = key + str(i)
                module.weight.data = torch.FloatTensor(ca_weights[key_string + "/kernel:0"])
                # module.bias.data = torch.FloatTensor(ca_weights[key_string + "/bias:0"])
                i += 1
                print("key: {}".format(key_string))
                if(is_cuda):
                    module.weight.data = module.weight.data.cuda()
                    
        # Upsampling
        key = "allconv"
        for module in self.ca_module.refine_decoder:
            if isinstance(module, nn.Conv2d):
                key_string = key + str(i)
                if(i == 13 or i == 15):
                    key_string = "{}_upsample/{}_upsample_conv".format(key_string, key_string)
                module.weight.data = torch.FloatTensor(ca_weights[key_string + "/kernel:0"])
                # module.bias.data = torch.FloatTensor(ca_weights[key_string + "/bias:0"])
                print("key: {}".format(key_string))
                i += 1
                if(is_cuda):
                    module.weight.data = module.weight.data.cuda()
                    
        
    def forward(self, objs, triples, obj_to_img=None, ca_masks=None, boxes_gt=None, masks_gt=None):
        # Forward to the Scene Graph Model
        img, boxes_pred, masks_pred, rel_scores = self.sg_model(
            objs, triples, obj_to_img, boxes_gt, masks_gt)

        # Build masks for the Contextual Attention
        # Input for CAModel: [SGModel out, ones, masks]
        # masks = utils.build_masks(img, f_boxes, f_obj_to_img)
        # masks = utils.to_var(masks, volatile = volatile)
        masks = ca_masks
        ones = torch.ones(img.shape[0], 1, img.shape[2], img.shape[3])
        ones = utils.to_var(ones)
        
        # need to downsample masks because CA module works on encoded image
        res_masks = F.interpolate(masks, scale_factor = 0.25, mode = 'nearest')

        refine_in = torch.cat([img, ones, masks], dim = 1)

        refine_img, _ = self.ca_module(refine_in, res_masks)
        
        return refine_img, boxes_pred, masks_pred, rel_scores

class ContextualSceneCritic(nn.Module):
    def __init__(self):
        super(ContextualSceneCritic, self).__init__()

        self.local_critic = ContextualCritic(is_local = True)
        self.global_critic = ContextualCritic(is_local = False)

    def forward(self, local_x, global_x, f_obj_to_img):
        local_out = self.local_critic(local_x, f_obj_to_img)
        global_out = self.global_critic(global_x)

        return local_out, global_out