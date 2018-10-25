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

# from model import Sg2ImModel
from cont_model import ContextualSceneModel, ContextualSceneCritic
from scene_graph.discriminators import PatchDiscriminator, AcCropDiscriminator
from losses import get_gan_losses, random_interpolate


class Solver(object):
    DEFAULTS = {}

    def __init__(self, vocab, foreground_objs, train_loader, test_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Load Vocab for model and data loader
        self.vocab = vocab
        self.foreground_objs = foreground_objs['foreground_idx']
        print("Foreground Objects: " + str(len(foreground_objs)))

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):

        # Generator = Scene Graph Model + Contextual Attention Module
        kwargs = {
            'foreground_objs': self.foreground_objs,
            'vocab': self.vocab,
            'image_size': self.image_size,
            'embedding_dim': self.embedding_dim,
            'gconv_dim': self.gconv_dim,
            'gconv_hidden_dim': self.gconv_hidden_dim,
            'gconv_num_layers': self.gconv_num_layers,
            'mlp_normalization': self.mlp_normalization,
            'refinement_network_dims': self.refinement_network_dims,
            'normalization': self.normalization,
            'activation': self.activation,
            'mask_size': self.mask_size,
            'layout_noise_dim': self.layout_noise_dim
        }

        self.generator = ContextualSceneModel(**kwargs)

        # Discriminators
        # OBJ Discriminator
        self.obj_discriminator = AcCropDiscriminator(vocab=self.vocab,
                                                     arch=self.d_obj_arch,
                                                     normalization=self.d_normalization,
                                                     activation=self.d_activation,
                                                     padding=self.d_padding,
                                                     object_size=self.crop_size)

        # IMG Discriminator
        self.img_discriminator = PatchDiscriminator(arch=self.d_img_arch,
                                                    normalization=self.d_normalization,
                                                    activation=self.d_activation,
                                                    padding=self.d_padding)

        # Local Critic
        self.critic = ContextualSceneCritic()

        # Loss
        self.gan_g_loss, self.gan_d_loss = get_gan_losses(self.gan_loss_type)
        self.critic_g_loss, self.critic_d_loss = get_gan_losses(
            self.critic_loss_type)

        # Mode of Model
        if self.init_iterations >= self.eval_mode_after:
            print("Setting Generator to Eval Mode")
            self.generator.eval()
        else:
            print("Setting Generator to Train Mode")
            self.generator.train()

        self.obj_discriminator.train()
        self.img_discriminator.train()
        self.critic.train()

        # Optimizers
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate)
        self.dis_obj_optimizer = torch.optim.Adam(
            self.obj_discriminator.parameters(), lr=self.learning_rate)
        self.dis_img_optimizer = torch.optim.Adam(
            self.img_discriminator.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate)

        # Print networks
        self.print_network(self.generator, 'Generator')
        self.print_network(self.obj_discriminator, 'Object Discriminator')
        self.print_network(self.img_discriminator, 'Image Discriminator')
        self.print_network(self.critic, 'Critic')

        if torch.cuda.is_available():
            self.generator.cuda()
            self.obj_discriminator.cuda()
            self.img_discriminator.cuda()
            self.critic.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.generator.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.img_discriminator.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D_IMG.pth'.format(self.pretrained_model))))
        self.obj_discriminator.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D_OBJ.pth'.format(self.pretrained_model))))
        self.img_discriminator.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_C.pth'.format(self.pretrained_model))))

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.generator.zero_grad()
        self.img_discriminator.zero_grad()
        self.obj_discriminator.zero_grad()
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
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0
        fixed_batch = self.build_sample_images(self.test_loader)

        # Start training
        iter_ctr = self.init_iterations
        e = iter_ctr // iters_per_epoch
        start_time = time.time()
        while True:
            # Stop training if iter_ctr reached num_iterations
            if iter_ctr >= self.num_iterations:
                break
            e += 1

            for i, batch in enumerate(tqdm(self.train_loader)):
                if iter_ctr == self.eval_mode_after:
                    self.generator.test()
                    self.gen_optimizer = torch.optim.Adam(
                        self.generator.parameters(), lr=self.learning_rate)

                masks = None
                if len(batch) == 6:
                    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
                elif len(batch) == 7:
                    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch

                start = time.time()

                # Prepare Data
                imgs = to_var(imgs)
                objs = to_var(objs)
                boxes = to_var(boxes)
                triples = to_var(triples)
                obj_to_img = to_var(obj_to_img)
                triple_to_img = to_var(triple_to_img)
                if(masks):
                    masks = to_var(masks)
                predicates = triples[:, 1]  # get p from triples(s, p ,o)
                
                # variables needed for generator and discriminator steps
                step_vars = (imgs, objs, boxes, obj_to_img, predicates, masks)

                # Foreground objects
                f_inds = [i for i, obj in enumerate(
                    objs) if obj in self.foreground_objs]
                f_boxes = boxes[f_inds]
                f_obj_to_img = obj_to_img[f_inds]

                # Logging
                loss = {}

                # Forward to Model
                model_boxes = boxes
                model_masks = masks
                model_out = self.generator(
                    objs, triples, obj_to_img, f_obj_to_img, f_boxes, boxes_gt=model_boxes, masks_gt=model_masks)

                imgs_pred, _, _, _ = model_out

                # build input for critics
                fake_local_patches = extract_patches(
                    imgs_pred, f_boxes, f_obj_to_img, self.patch_size, self.batch_size)
                real_local_patches = extract_patches(
                    imgs, f_boxes, f_obj_to_img, self.patch_size, self.batch_size)

                # Forward to Critic
                critic_pred = None
                if self.critic is not None:
                    local_vectors = torch.cat(
                        [fake_local_patches, real_local_patches], dim=0)
                    global_vectors = torch.cat([imgs_pred, imgs], dim=0)

                    # Feed input to the critic to get scores
                    local_critic_out, global_critic_out = self.critic(
                        local_vectors, global_vectors, f_obj_to_img, self.batch_size)

                    # Split scores to (fake, real)
                    fake_local_pred, real_local_pred = torch.split(
                        local_critic_out, self.batch_size, dim=0)
                    fake_global_pred, real_global_pred = torch.split(
                        global_critic_out, self.batch_size, dim=0)

                    critic_pred = (fake_local_pred, real_local_pred, fake_global_pred, real_global_pred)
                    # for gradient penalty of critic
                    gp_vectors = (real_local_patches, fake_local_patches)

                # Generator Step
                total_loss, losses = self.generator_step(step_vars, model_out, critic_pred)

                loss['G/total_loss'] = total_loss.data.item()
                loss = self.log_losses(loss, 'G', losses)

                # Discriminator Step
                total_loss, dis_obj_losses, dis_img_losses, critic_losses = self.discriminator_step(
                    imgs_pred, step_vars, critic_pred, gp_vectors)

                loss['D/total_loss'] = total_loss.data.item()
                loss = self.log_losses(loss, 'D', dis_obj_losses)
                loss = self.log_losses(loss, 'D', dis_img_losses)

                # Print out log info
                if (i + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = (self.num_iterations -
                                  iter_ctr + 1) * elapsed / (iter_ctr + 1)
                    epoch_time = (iters_per_epoch - i) * elapsed / \
                        (iter_ctr + 1)

                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed {}/{} -- {} , Iteration [{}/{}], Epoch [{}]".format(
                        elapsed, epoch_time, total_time, iter_ctr + 1, self.num_iterations, e)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(
                                tag, value, iter_ctr + 1)

                # Save model checkpoints
                if (iter_ctr + 1) % self.model_save_step == 0:
                    torch.save(self.generator.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_{}_G.pth'.format(iter_ctr + 1, e, i + 1)))
                    torch.save(self.obj_discriminator.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_{}_D_OBJ.pth'.format(iter_ctr + 1, e, i + 1)))
                    torch.save(self.img_discriminator.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_{}_D_IMG.pth'.format(iter_ctr + 1, e, i + 1)))

                if (iter_ctr + 1) % self.sample_step == 0:
                    self.sample_images(fixed_batch, e, i)

                iter_ctr += 1

    def discriminator_step(self, imgs_pred, step_vars, critic_pred, gp_vectors):
        ac_loss_real = None
        ac_loss_fake = None
        d_obj_losses = None
        d_img_losses = None
        
        imgs, objs, boxes, obj_to_img, _, _ = step_vars
        # Step for Obj Discriminator
        if self.obj_discriminator is not None:
            d_obj_losses = LossManager()
            imgs_fake = imgs_pred.detach()
            scores_fake, ac_loss_fake = self.obj_discriminator(
                imgs_fake, objs, boxes, obj_to_img)
            scores_real, ac_loss_real = self.obj_discriminator(
                imgs, objs, boxes, obj_to_img)

            d_obj_gan_loss = self.gan_d_loss(scores_real, scores_fake)
            d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
            d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
            d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

            self.reset_grad()
            d_obj_losses.total_loss.backward()
            self.dis_obj_optimizer.step()

        # Step for Img Discriminator
        if self.img_discriminator is not None:
            d_img_losses = LossManager()
            imgs_fake = imgs_pred.detach()
            scores_fake = self.img_discriminator(imgs_fake)
            scores_real = self.img_discriminator(imgs)

            d_img_gan_loss = self.gan_d_loss(scores_real, scores_fake)
            d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')

            self.reset_grad()
            d_img_losses.total_loss.backward()
            self.dis_img_optimizer.step()

        if critic_pred is not None:
            # critic pred: (fake local, real local, fake global, real global)
            fake_local_pred, real_local_pred, fake_global_pred, real_global_pred = critic_pred
            critic_losses = LossManager()
            # Local Loss
            local_loss = self.critic_d_loss(real_local_pred, fake_local_pred)
            critic_losses.add_loss(local_loss, 'c_local_loss')

            # Global Loss
            global_loss = self.critic_d_loss(real_global_pred, fake_global_pred)
            critic_losses.add_loss(global_loss, 'c_global_loss')

            if gp_vectors is not None:
                # Gradient Penalty Loss
                real_local_patches, fake_local_patches = gp_vectors
                # Interpolate thingy here
                print("LOCAL: ")
                local_interpolate = random_interpolate(real_local_patches, fake_local_patches)
                print("GLOBAL: ")
                global_interpolate = random_interpolate(imgs, imgs_pred)
                # GP Loss Here

            self.reset_grad()
            critic_losses.total_loss.backward()
            self.critic_optimizer.step()

        total_loss = d_obj_losses.total_loss + \
            d_img_losses.total_loss + critic_losses.total_loss
        return total_loss, d_obj_losses, d_img_losses, critic_losses

    def generator_step(self, step_vars, model_out, critic_pred=None):
        losses = {}

        imgs, objs, boxes, obj_to_img, predicates, masks = step_vars
        imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out

        total_loss = torch.zeros(1).to(imgs)
        skip_pixel_loss = (boxes is None)
        # Pixel Loss
        l1_pixel_weight = self.l1_pixel_loss_weight
        if skip_pixel_loss:
            l1_pixel_weight = 0

        # Local patches for hand picked foreground objects
        print("OBJS: " + str(objs))

        l1_pixel_loss = F.l1_loss(imgs_pred, imgs)

        total_loss = self.add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                                   l1_pixel_weight)

        # Box Loss
        loss_bbox = F.mse_loss(boxes_pred, boxes)
        total_loss = self.add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                                   self.bbox_pred_loss_weight)

        if self.predicate_pred_loss_weight > 0:
            loss_predicate = F.cross_entropy(predicate_scores, predicates)
            total_loss = self.add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                                       self.predicate_pred_loss_weight)

        if self.mask_loss_weight > 0 and masks is not None and masks_pred is not None:
            # Mask Loss
            mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
            total_loss = self.add_loss(total_loss, mask_loss, losses, 'mask_loss',
                                       self.mask_loss_weight)

        if self.obj_discriminator is not None:
            # OBJ AC Loss: Classification of Objects
            scores_fake, ac_loss = self.obj_discriminator(
                imgs_pred, objs, boxes, obj_to_img)
            total_loss = self.add_loss(total_loss, ac_loss, losses, 'ac_loss',
                                       self.ac_loss_weight)

            # OBJ GAN Loss: Real vs Fake
            weight = self.discriminator_loss_weight * self.d_obj_weight
            total_loss = self.add_loss(total_loss, self.gan_g_loss(scores_fake), losses,
                                       'g_gan_obj_loss', weight)

        if self.img_discriminator is not None:
            # IMG GAN Loss: Patches should be realistic
            scores_fake = self.img_discriminator(imgs_pred)
            weight = self.discriminator_loss_weight * self.d_img_weight
            total_loss = self.add_loss(total_loss, self.gan_g_loss(scores_fake), losses,
                                       'g_gan_img_loss', weight)

        if critic_pred is not None:
            # critic pred: (fake local, real local, fake global, real global)
            fake_local_pred, _, fake_global_pred, _ = critic_pred
            
            # Local Patch Loss
            local_loss = self.critic_g_loss(fake_local_pred)
            print("LOCAL LOSS: " + str(local_loss))
            total_loss = self.add_loss(total_loss, local_loss, losses,
                                       'g_local_loss')

            # Global Loss
            global_loss = self.critic_g_loss(fake_global_pred)
            total_loss = self.add_loss(total_loss, global_loss, losses,
                                       'g_global_loss')

        losses['total_loss'] = total_loss.item()

        self.reset_grad()
        total_loss.backward(retain_graph = True)
        # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
        self.gen_optimizer.step()

        return total_loss, losses

    def sample_images(self, batch, e=1, i=1):

        self.generator.eval()
        objs, triples, obj_to_img, f_obj_to_img, f_boxes = batch

        objs = to_var(objs, volatile=True)
        triples = to_var(triples, volatile=True)
        obj_to_img = to_var(obj_to_img, volatile=True)

        fake_images, _, _, _ = self.generator(
            objs, triples, obj_to_img, f_obj_to_img=f_obj_to_img, f_boxes=f_boxes, mode='test')

        save_image(imagenet_deprocess_batch(fake_images, convert_range=False),
                   os.path.join(self.sample_path, '{}_{}_fake.png'.format(e, i + 1)), nrow=1, padding=0)
        print('Translated images and saved into {}..!'.format(self.sample_path))

    def build_sample_images(self, data_loader):
        batch = next(iter(data_loader))

        if len(batch) == 6:
            _, objs, boxes, triples, obj_to_img, _ = batch
        elif len(batch) == 7:
            _, objs, boxes, _, triples, obj_to_img, _ = batch

        f_inds = [i for i, obj in enumerate(
            objs) if obj in self.foreground_objs]
        f_boxes = boxes[f_inds]
        f_obj_to_img = obj_to_img[f_inds]

        return (objs, triples, obj_to_img, f_obj_to_img, f_boxes)
