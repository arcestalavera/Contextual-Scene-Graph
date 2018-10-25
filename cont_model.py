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
                                           refinement_dims=kwargs['refinement_network_dims'],
                                           normalization=kwargs['normalization'],
                                           activation=kwargs['activation'],
                                           mask_size=kwargs['mask_size'],
                                           layout_noise_dim=kwargs['layout_noise_dim'])

        # handle refinement network arguments
        self.ca_module = CAModel.RefineNetwork()

        # self.foreground_objs = kwargs['foreground_objs']

    def forward(self, objs, triples, obj_to_img=None, f_obj_to_img=None,
                f_boxes=None, boxes_gt=None, masks_gt=None, mode = 'train'):
        
        if(mode == 'train'):
            volatile = False
        else:
            volatile = True

        # Forward to the Scene Graph Model
        img, boxes_pred, masks_pred, rel_scores = self.sg_model(
            objs, triples, obj_to_img, boxes_gt, masks_gt)

        # Build masks for the Contextual Attention
        # Input for CAModel: [SGModel out, ones, masks]
        masks = utils.build_masks(img, f_boxes, f_obj_to_img)
        masks = utils.to_var(masks, volatile = volatile)
        
        ones = torch.ones(img.shape[0], 1, img.shape[2], img.shape[3])
        ones = utils.to_var(ones, volatile = volatile)
        
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

    def forward(self, local_x, global_x, f_obj_to_img, batch_size):
        local_out = self.local_critic(local_x, True, f_obj_to_img, batch_size)
        global_out = self.global_critic(global_x, False)

        print("GLOBAL: " + str(global_out.shape))

        return local_out, global_out