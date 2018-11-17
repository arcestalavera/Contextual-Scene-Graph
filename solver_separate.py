import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torchvision.utils import save_image
from utils import *
from data_loader import *
from tqdm import tqdm
from imageio import imwrite

# from model import Sg2ImModel
from cont_model import ContextualSceneModel, ContextualSceneCritic
from scene_graph.discriminators import PatchDiscriminator, AcCropDiscriminator
from scene_graph.bilinear import crop_bbox_batch
from scene_graph.model import Sg2ImModel
import contextual_attention.model as CAModel
from losses import get_gan_losses, random_interpolate, spatial_l1

class Solver(object):
    DEFAULTS = {}

    def __init__(self, vocab, foreground_objs, ca_weights, train_loader, test_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Load Vocab for model and data loader
        self.vocab = vocab
        self.foreground_objs = foreground_objs['foreground_idx']
        self.ca_weights = ca_weights

        # Build tensorboard if use
        self.load_generator()
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):

        # Generator = Scene Graph Model + Contextual Attention Module
        # kwargs = {
        #     'foreground_objs': self.foreground_objs,
        #     'vocab': self.vocab,
        #     'image_size': self.image_size,
        #     'embedding_dim': self.embedding_dim,
        #     'gconv_dim': self.gconv_dim,
        #     'gconv_hidden_dim': self.gconv_hidden_dim,
        #     'gconv_num_layers': self.gconv_num_layers,
        #     'mlp_normalization': self.mlp_normalization,
        #     'refinement_network_dims': self.refinement_network_dims,
        #     'normalization': self.normalization,
        #     'activation': self.activation,
        #     'mask_size': self.mask_size,
        #     'layout_noise_dim': self.layout_noise_dim,
        #     # 'ca_weights': self.ca_weights,
        # }

        # self.generator = Sg2ImModel(**kwargs)        
        
        # refinement network
        self.refinement = CAModel.RefineNetwork()

        self.critic = ContextualSceneCritic()

        if(self.ca_weights):
            self.refinement.load_ca_weights(self.ca_weights)

        # Losses (WGAN-GP for the refinement network)
        self.critic_g_loss, self.critic_d_loss = get_gan_losses(
            self.critic_gan_loss)
        self.critic_gp_loss = get_gan_losses(self.critic_gp_loss)

        # We only need to train the refinement network and the critic
        self.generator.eval()
        self.refinement.train()
        self.critic.train()

        # Optimizers
        self.refinement_optimizer = torch.optim.Adam(
            self.refinement.parameters(), lr=self.learning_rate, betas=(self.optim_beta1, self.optim_beta2))

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate, betas=(self.optim_beta1, self.optim_beta2))

        # Print networks
        # self.print_network(self.generator, 'Generator')
        # self.print_network(self.refinement, 'Refinement Module')
        # self.print_network(self.critic, 'Critic')

        if torch.cuda.is_available():
            self.generator.cuda()
            self.refinement.cuda()
            self.critic.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_generator(self):
        checkpoint = torch.load(os.path.join(self.model_save_path, 'generator/{}.pt'.format(
            self.generator_model)))
        self.generator = Sg2ImModel(**checkpoint['model_kwargs'])
        self.generator.load_state_dict(checkpoint['model_state'])
        # self.generator.load_state_dict(torch.load(os.path.join(
        #     self.model_save_path, '{}_G.pth'.format(self.generator_model))))
        print("loaded trained generator!")
        
    def load_pretrained_model(self):
        self.refinement.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_R.pth'.format(self.pretrained_model))))
        self.critic.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_C.pth'.format(self.pretrained_model))))

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.refinement.zero_grad()
        self.critic.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def lr_scheduler(self, lr_sched):
        new_lr = lr_sched.compute_lr()
        self.update_lr(new_lr)

    def add_loss(self, total_loss, curr_loss, loss_dict, loss_name, weight=1):
        curr_loss = curr_loss * weight
        loss_dict[loss_name] = curr_loss.item()
        if total_loss is not None:
            total_loss += curr_loss
        else:
            total_loss = curr_loss
        return total_loss

    def log_losses(self, loss, module, losses):
        for loss_name, loss_val in losses.items():
            # print("LOSS {}: {}".format(loss_name,loss_val))
            loss["{}/{}".format(module, loss_name)] = loss_val

        return loss

    def train(self):
        iters_per_epoch = len(self.train_loader)
        print("Iterations per epoch: " + str(iters_per_epoch))

        # Start with trained model if exists
        fixed_batch = self.build_sample_images(self.test_loader)

        # Start training
        # e = iter_ctr // iters_per_epoch
        e = self.init_epoch
        iter_ctr = e * iters_per_epoch
        print("iter_ctr: {}".format(iter_ctr))
        start_time = time.time()
        for e in range(self.num_epochs):
            for i, batch in enumerate(tqdm(self.train_loader)):
                start = time.time()
                masks = None
                if len(batch) == 6:
                    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
                elif len(batch) == 7:
                    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch

                # to_var the inputs
                imgs = to_var(imgs)
                objs = to_var(objs)
                boxes = to_var(boxes)
                if masks:
                    masks = to_var(masks)
                triples = to_var(triples)
                obj_to_img = to_var(obj_to_img)
                triple_to_img = to_var(triple_to_img)
                predicates = triples[:, 1]  # get p from triples(s, p ,o)
                
                # Forward to Model
                model_boxes = boxes
                model_masks = masks
                with torch.no_grad():
                    gen_out, boxes_pred, masks_pred, rel_scores = self.generator(
                        objs, triples, obj_to_img, boxes_gt=model_boxes, masks_gt=model_masks)
                    
                # Foreground objects
                f_inds = [i for i, obj in enumerate(
                    objs) if obj in self.foreground_objs]
                f_boxes = boxes_pred[f_inds]
                f_obj_to_img = obj_to_img[f_inds]

                # Prepare Input for CA Module
                # Build Masks for Contextual Attention Module
                ca_masks = build_masks(gen_out, f_boxes, f_obj_to_img)
                ca_masks = to_var(ca_masks)

                ones = torch.ones(gen_out.shape[0], 1, gen_out.shape[2], gen_out.shape[3])
                ones = to_var(ones)
                
                # need to downsample masks because CA module works on encoded image
                res_masks = F.interpolate(ca_masks, scale_factor = 0.25, mode = 'nearest')
                refine_in = torch.cat([gen_out, ones, ca_masks], dim = 1)

                # feed to the refinement network
                imgs_pred, _ = self.refinement(refine_in, res_masks)
                foreground = (f_boxes, f_obj_to_img)

                # get losses then back prop
                refinement_losses = self.refinement_step(imgs, imgs_pred, foreground, ca_masks)
                
                critic_losses = self.critic_step(imgs, imgs_pred, foreground, ca_masks)
         
                # Logging
                loss = {}

                loss['R/total_loss'] = refinement_losses.total_loss
                loss = self.log_losses(loss, 'R', refinement_losses)
                loss['C/total_loss'] = critic_losses.total_loss
                loss = self.log_losses(loss, 'C', critic_losses)

                # Print out log info
                if (i + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                    epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                    
                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Save model checkpoints
                if (iter_ctr + 1) % self.model_save_step == 0:
                    torch.save(self.refinement.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_R.pth'.format(e + 1, i + 1)))
                    torch.save(self.critic.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_C.pth'.format(e + 1, i + 1)))

                if (iter_ctr + 1) % self.sample_step == 0:
                    self.sample_images(fixed_batch, iter_ctr)

                iter_ctr += 1

    def refinement_step(self, imgs, imgs_pred, foreground, ca_masks):
        # Generator (Refinement Network Loss)
        refinement_losses = LossManager()
        # unpack
        f_boxes, f_obj_to_img = foreground
        fake_local_patches = crop_bbox_batch(imgs_pred, f_boxes, f_obj_to_img, self.patch_size)
        real_local_patches = crop_bbox_batch(imgs, f_boxes, f_obj_to_img, self.patch_size)

        # Get Spatial Discounted L1 Loss
        spatial_loss = spatial_l1(fake_local_patches, real_local_patches, f_obj_to_img, self.patch_size, gamma = self.spatial_gamma)
        refinement_losses.add_loss(self.spatial_loss_weight * spatial_loss, 'r_spatial_loss')

        # Get L1 Pix Loss
        l1_pixel_weight = self.l1_pixel_loss_weight
        l1_pixel_loss = F.l1_loss(imgs_pred, imgs)
        refinement_losses.add_loss(self.l1_pixel_loss_weight * l1_pixel_loss, 'r_pix_loss')

        # Get WGAN Loss
        fake_local_pred, fake_global_pred = self.critic(fake_local_patches, imgs_pred, f_obj_to_img, get_ave = True)
        
        # Local Patch Loss
        local_loss = self.critic_g_loss(fake_local_pred)

        # Global Loss
        global_loss = self.critic_g_loss(fake_global_pred)

        critic_loss = self.critic_global_weight * global_loss + local_loss
        refinement_losses.add_loss(self.critic_g_weight * critic_loss, 'r_wgan_loss')

        # backward
        self.reset_grad()
        refinement_losses.total_loss.backward()
        self.refinement_optimizer.step()
        return refinement_losses
            
    def critic_step(self, imgs, imgs_pred, foreground, ca_masks):
        # detatch img_pred and ca_masks so refinement wouldn't be updated
        critic_losses = LossManager()
        imgs_fake = imgs_pred.detach()
        f_boxes, f_obj_to_img = foreground

        patch_masks = crop_bbox_batch(
                    ca_masks, f_boxes, f_obj_to_img, self.patch_size)

        # build input for critics
        fake_local_patches = crop_bbox_batch(
            imgs_fake, f_boxes, f_obj_to_img, self.patch_size)
        real_local_patches = crop_bbox_batch(
            imgs, f_boxes, f_obj_to_img, self.patch_size)
        local_vectors = torch.cat(
            [fake_local_patches, real_local_patches], dim=0)
        
        global_vectors = torch.cat([imgs_fake, imgs], dim=0)

        # Feed to the critic then split output to (fake, real)
        local_critic_out, global_critic_out = self.critic(
            local_vectors, global_vectors, f_obj_to_img)
        
        fake_local_pred, real_local_pred = torch.split(
            local_critic_out, imgs.shape[0], dim=0)
        fake_global_pred, real_global_pred = torch.split(
            global_critic_out, imgs.shape[0], dim=0)
        
        # Local Loss
        local_loss = self.critic_d_loss(real_local_pred, fake_local_pred)

        # Global Loss
        global_loss = self.critic_d_loss(real_global_pred, fake_global_pred)

        critic_losses.add_loss(global_loss + local_loss, 'c_wgan_loss')

        # Gradient Penalty Loss
        # real_local_patches, fake_local_patches = gp_vectors
        # global_masks, local_masks = gp_masks

        # Interpolate Images
        local_interpolate = random_interpolate(
            real_local_patches, fake_local_patches)
        global_interpolate = random_interpolate(imgs, imgs_fake)

        local_interpolate = to_var(local_interpolate, requires_grad = True)
        global_interpolate = to_var(global_interpolate, requires_grad = True)
        
        # GP Loss
        # Forward interpolated images to the critic
        local_gp_out, global_gp_out = self.critic(
            local_interpolate, global_interpolate, f_obj_to_img)

        local_gp = self.critic_gp_loss(
            local_interpolate, local_gp_out, mask=patch_masks, f_obj_to_img=f_obj_to_img)
        global_gp = self.critic_gp_loss(
            global_interpolate, global_gp_out, mask=ca_masks)

        critic_losses.add_loss(
            self.critic_gp_weight * (local_gp + global_gp), 'c_gp_loss')

        # backpropagate and optimize critic
        self.reset_grad()
        critic_losses.total_loss.backward()
        self.critic_optimizer.step()
        return critic_losses

    def sample_images(self, batch, iter_ctr=1):
        self.refinement.eval()
        refine_in, res_masks = batch

        with torch.no_grad():
            fake_images, _ = self.refinement(refine_in, res_masks)

        save_image(imagenet_deprocess_batch(fake_images, convert_range=False),
                   os.path.join(self.sample_path, '{}_fake_refined.png'.format(iter_ctr + 1)), nrow=1, padding=0)
        print('Translated images and saved into {}..!'.format(self.sample_path))

    def build_sample_images(self, data_loader):
        self.generator.eval()
        batch = next(iter(data_loader))

        if len(batch) == 6:
            imgs, objs, boxes, triples, obj_to_img, _ = batch
        elif len(batch) == 7:
            imgs, objs, boxes, _, triples, obj_to_img, _ = batch

        with torch.no_grad():
            objs = to_var(objs)
            triples = to_var(triples)
            obj_to_img = to_var(obj_to_img)
            
            gen_out, boxes_pred, _, _ = self.generator(
                objs, triples, obj_to_img)
            
            save_image(imagenet_deprocess_batch(gen_out, convert_range=False),
                   os.path.join(self.sample_path, 'fake_unrefined.png'), nrow=1, padding=0)

            f_inds = [i for i, obj in enumerate(
                objs) if obj in self.foreground_objs]
            f_boxes = boxes_pred[f_inds]
            f_obj_to_img = obj_to_img[f_inds]

            ca_masks = build_masks(imgs, f_boxes, f_obj_to_img)
            ca_masks = to_var(ca_masks)

            ones = torch.ones(gen_out.shape[0], 1, gen_out.shape[2], gen_out.shape[3])
            ones = to_var(ones)
                        
            # need to downsample masks because CA module works on encoded image
            res_masks = F.interpolate(ca_masks, scale_factor = 0.25, mode = 'nearest')
            refine_in = torch.cat([gen_out, ones, ca_masks], dim = 1)

        return refine_in, res_masks

    def test(self, num_batch=None):
        self.generator.eval()
        self.refinement.eval()

        if num_batch != None:
            stop_test = num_batch
        else:
            stop_test = len(self.test_loader)

        img_num = 0
        graph_num = 0
        for i, batch in enumerate(self.test_loader):
            if len(batch) == 6:
                imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
            elif len(batch) == 7:
                imgs, objs, boxes, _, triples, obj_to_img, triple_to_img = batch

            with torch.no_grad():
                objs = to_var(objs)
                triples = to_var(triples)
                obj_to_img = to_var(obj_to_img)

                gen_out, boxes_pred, _, _ = self.generator(
                        objs, triples, obj_to_img)

                f_inds = [i for i, obj in enumerate(
                    objs) if obj in self.foreground_objs]
                f_boxes = boxes_pred[f_inds]
                f_obj_to_img = obj_to_img[f_inds]

                # Prepare Input for CA Module
                # Build Masks for Contextual Attention Module
                ca_masks = build_masks(gen_out, f_boxes, f_obj_to_img)
                ca_masks = to_var(ca_masks)

                ones = torch.ones(gen_out.shape[0], 1, gen_out.shape[2], gen_out.shape[3])
                ones = to_var(ones)
                
                # need to downsample masks because CA module works on encoded image
                res_masks = F.interpolate(ca_masks, scale_factor = 0.25, mode = 'nearest')
                refine_in = torch.cat([gen_out, ones, ca_masks], dim = 1)

                # feed to the refinement network
                fake_images, attn_out = self.refinement(refine_in, res_masks)

            # Save Images
            fake_images = imagenet_deprocess_batch(
                fake_images, convert_range=True)
            for j in range(fake_images.shape[0]):
                img = fake_images[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(
                    self.test_path, "fake_{}.png".format(img_num))
                imwrite(img_path, img)
                img_num += 1

            graph_num = self.export_relationship_graph(
                objs, triples, triple_to_img, graph_num)
            if(i + 1 >= stop_test):
                break

        print('Translated images and saved into {}..!'.format(self.test_path))

    def export_relationship_graph(self, objs, triples, triple_to_img, graph_num):
        obj_names = self.vocab['object_idx_to_name']
        pred_names = self.vocab['pred_idx_to_name']

        # Do it per imag
        with open(os.path.join(self.test_path, 'test.txt'), 'a') as graph_file:
            for i in range(max(triple_to_img) + 1):
                print("===================================", file=graph_file)
                print("image: {}".format(graph_num), file=graph_file)
                triple_inds = (triple_to_img == i).nonzero()
                img_triples = triples[triple_inds].view(-1, 3)
                # print("triples: {}".format(img_triples))
                for s, p, o in img_triples:
                    if(pred_names[p] == '__in_image__'):
                        continue
                    s_label = obj_names[objs[s]]
                    p_label = pred_names[p]
                    o_label = obj_names[objs[o]]
                    print("{} --- {} ---> {}".format(s_label,
                                                     p_label, o_label), file=graph_file)
                graph_num += 1

        return graph_num
