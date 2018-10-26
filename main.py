import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
from datetime import datetime
import zipfile
import json
import utils

def zipdir(path, ziph):
    files = os.listdir(path)
    for file in files:
        if file.endswith(".py") or file.endswith("cfg"):
            ziph.write(os.path.join(path, file))
            if file.endswith("cfg"):
                os.remove(file)

def save_config(config):
    CONFIG_DIR = './config'
    utils.mkdir(CONFIG_DIR)
    current_time = str(datetime.now()).replace(":", "_")
    save_name = os.path.join(CONFIG_DIR, "txt2img_files_{}.{}")
    with open(save_name.format(current_time, "cfg"), "w") as f:
        for k, v in sorted(args.items()):
            f.write('%s: %s\n' % (str(k), str(v)))

    zipf = zipfile.ZipFile(save_name.format(
        current_time, "zip"), 'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)
    mkdir(config.sample_path)

    # Load Vocab and Handpicked Foreground Objects
    vocab = json.load(open(config.vocab_json, 'r'))
    foreground_objs = json.load(open(config.foreground_json, 'r'))

    # Data Loaders
    H, W = config.image_size
    train_loader = get_loader(vocab, config.images_path, config.train_h5,
                              config.batch_size, image_size = (H, W), mode = 'train')
    test_loader = get_loader(vocab, config.images_path, config.test_h5,
                            config.batch_size, image_size = (H, W), mode = 'test')

    # Solver
    solver = Solver(vocab, foreground_objs, train_loader, test_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Variables
    VG_DIR = 'vg'

    # Scene Graph Model Generator
    # Set this to 0 to use no masks
    parser.add_argument('--mask_size', default=16, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--gconv_dim', default=128, type=int)
    parser.add_argument('--gconv_hidden_dim', default=512, type=int)
    parser.add_argument('--gconv_num_layers', default=5, type=int)
    parser.add_argument('--mlp_normalization', default='none', type=str)
    parser.add_argument('--refinement_network_dims',
                        default='1024,512,256,128,64', type=int_tuple)
    parser.add_argument('--normalization', default='batch')
    parser.add_argument('--activation', default='leakyrelu-0.2')
    parser.add_argument('--layout_noise_dim', default=32, type=int)
    parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

    # Generator losses
    parser.add_argument('--mask_loss_weight', default=0, type=float)
    parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
    parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
    parser.add_argument('--predicate_pred_loss_weight',
                        default=0, type=float)  # DEPRECATED

    # Scene Graph Model Discriminators
    # Generic discriminator options
    parser.add_argument('--discriminator_loss_weight',
                        default=0.01, type=float)
    parser.add_argument('--gan_loss_type', default='gan')
    parser.add_argument('--d_clip', default=None, type=float)
    parser.add_argument('--d_normalization', default='batch')
    parser.add_argument('--d_padding', default='valid')
    parser.add_argument('--d_activation', default='leakyrelu-0.2')

    # Object discriminator
    parser.add_argument('--d_obj_arch',
                        default='C4-64-2,C4-128-2,C4-256-2')
    parser.add_argument('--crop_size', default=32, type=int)
    parser.add_argument('--d_obj_weight', default=1.0,
                        type=float)  # multiplied by d_loss_weight
    parser.add_argument('--ac_loss_weight', default=0.1, type=float)

    # Image discriminator
    parser.add_argument('--d_img_arch',
                        default='C4-64-2,C4-128-2,C4-256-2')
    parser.add_argument('--d_img_weight', default=1.0,
                        type=float)  # multiplied by d_loss_weight

    # Critics
    parser.add_argument('--critic_gan_loss', default='wgan')
    parser.add_argument('--critic_gp_loss', default='gp')
    parser.add_argument('--critic_g_weight', default=0.001, type=float)
    parser.add_argument('--critic_global_weight', default=1.0, type=float)
    parser.add_argument('--critic_gp_weight', default=10.0, type=float)
    
    # Dataset options common to both VG and COCO
    parser.add_argument('--image_size', default='64,64', type=int_tuple)
    parser.add_argument('--num_train_samples', default=None, type=int)
    parser.add_argument('--num_val_samples', default=1024, type=int)
    parser.add_argument('--shuffle_val', default=True, type=bool_flag)
    parser.add_argument('--include_relationships',
                        default=True, type=bool_flag)

    # VG-specific options
    parser.add_argument(
        '--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
    parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
    parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
    parser.add_argument('--test_h5', default=os.path.join(VG_DIR, 'test.h5'))
    parser.add_argument(
        '--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
    parser.add_argument('--max_objects_per_image', default=10, type=int)
    parser.add_argument('--vg_use_orphaned_objects',
                        default=True, type=bool_flag)
    parser.add_argument(
        '--foreground_json', default=os.path.join(VG_DIR, 'foreground.json'))

    # Training settings
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    # parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--init_iterations', type=int, default=0)
    parser.add_argument('--num_iterations', type=int, default=1000000)
    parser.add_argument('--eval_mode_after', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--patch_size', type=int, default=32) 
    
    # Misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join(VG_DIR, 'images'))
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')


    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=24947)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    save_config(config)
    main(config)
