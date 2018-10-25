#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import inspect
import subprocess
from contextlib import contextmanager
from torchvision import transforms as T
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scene_graph.bilinear import crop_bbox_batch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def extract_patches(imgs, boxes, obj_to_img, patch_size=32, batch_size = None):
  # Box Format: (x0 , y0,  x1, y1)
  img_patches = []

  # Utilize crop function of scene graph
  img_patches = crop_bbox_batch(imgs, boxes, obj_to_img, patch_size)

  # Check if some image does not have patches
  print("{} VS. {}".format(len(set(obj_to_img)), batch_size))
  if(len(set(obj_to_img)) != batch_size):
    print("I AM NOT COMPLETE")

  return img_patches


def build_masks(imgs, boxes, obj_to_img):
  H, W = imgs.shape[2], imgs.shape[3]
  masks = torch.zeros((imgs.shape[0], 1, H, W))
  
  # create masks
  print("MASK STATS")
  print("BOXES: " + str(boxes.shape))
  print("OBJ TO IMG: " + str(obj_to_img.shape))
  for i in range(boxes.shape[0]):
    img_ind = obj_to_img[i]
    
    box = torch.trunc(boxes[i] * H).int()
    masks[img_ind, :, box[1]:box[3] + 1, box[0]:box[2] + 1] = 1

  print("MASKS: " + str(masks.shape))
  return masks
  

def imagenet_deprocess(rescale_image=True):
  transforms = [
      T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
      T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(rescale)
  return T.Compose(transforms)


def rescale(x):
  lo, hi = x.min(), x.max()
  return x.sub(lo).div(hi - lo)


def imagenet_deprocess_batch(imgs, rescale=True, convert_range = True):
  """
  Input:
  - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

  Output:
  - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
    in the range [0, 255]
  """
  if isinstance(imgs, torch.autograd.Variable):
    imgs = imgs.data
  imgs = imgs.cpu().clone()
  deprocess_fn = imagenet_deprocess(rescale_image=rescale)
  imgs_de = []
  for i in range(imgs.size(0)):
    img_de = deprocess_fn(imgs[i])[None]
    if(convert_range):
      img_de = img_de.mul(255).clamp(0, 255).byte()
    imgs_de.append(img_de)
  imgs_de = torch.cat(imgs_de, dim=0)
  return imgs_de


def to_var(x, requires_grad=False, volatile=False):
  if torch.cuda.is_available():
    if(volatile):
      x = Variable(x, volatile=volatile).cuda()
    else:
      x = x.cuda()
  x.requires_grad_(requires_grad)
  return x


def mkdir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def int_tuple(s):
  return tuple(int(i) for i in s.split(','))


def float_tuple(s):
  return tuple(float(i) for i in s.split(','))


def str_tuple(s):
  return tuple(s.split(','))


def bool_flag(s):
  if s == '1':
    return True
  elif s == '0':
    return False
  msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
  raise ValueError(msg % s)


def lineno():
  return inspect.currentframe().f_back.f_lineno


def get_gpu_memory():
  torch.cuda.synchronize()
  opts = [
      'nvidia-smi', '-q', '--gpu=' + str(0), '|', 'grep', '"Used GPU Memory"'
  ]
  cmd = str.join(' ', opts)
  ps = subprocess.Popen(
      cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  output = ps.communicate()[0].decode('utf-8')
  output = output.split("\n")[1].split(":")
  consumed_mem = int(output[1].strip().split(" ")[0])
  return consumed_mem


@contextmanager
def timeit(msg, should_time=True):
  if should_time:
    torch.cuda.synchronize()
    t0 = time.time()
  yield
  if should_time:
    torch.cuda.synchronize()
    t1 = time.time()
    duration = (t1 - t0) * 1000.0
    print('%s: %.2f ms' % (msg, duration))


class LossManager(object):
  def __init__(self):
    self.total_loss = None
    self.all_losses = {}

  def add_loss(self, loss, name, weight=1.0):
    cur_loss = loss * weight
    if self.total_loss is not None:
      self.total_loss += cur_loss
    else:
      self.total_loss = cur_loss

    self.all_losses[name] = cur_loss.data.cpu().item()

  def items(self):
    return self.all_losses.items()
