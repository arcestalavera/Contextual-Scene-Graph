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

import argparse, json, os

from imageio import imwrite
import torch

from scene_graph.model import Sg2ImModel
from utils import imagenet_deprocess_batch
# import scene_graph.vis as vis

from scene_graph.bilinear import crop_bbox_batch
import json
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

def extract_patches(imgs, objs, boxes, obj_to_img, f_inds, patch_size=(32, 32)):
  H, W = imgs.shape[2], imgs.shape[3]
  # Box Format: (x0 , y0,  x1, y1)
  img_patches = []
  fore_objects = torch.cuda.LongTensor(f_inds['foreground_idx'])

  # use only boxes of hand picked foreground objects
  boxes_inds = [i for i, obj in enumerate(objs) if obj in fore_objects]

  print("BOXES TO KEEP: " + str(boxes_inds))
  print("MY BOXES: " + str(boxes))
  # extract image patches using the boxes
  for ind in boxes_inds:
    img_ind = obj_to_img[ind]

    box = torch.trunc(boxes[ind] * H).int()
    print("OBJ: " + str(objs))
    print("MY BOX: " + str(box))
    img_patch = imgs[img_ind, :, box[1]:box[3] + 1, box[0]:box[2] + 1]
    print("IMG PATCH: " + str(img_patch.shape))
    img_patches.append(img_patch)

  img_patches = torch.cat(img_patches)
  img_patches = F.interpolate(img_patches, size=patch_size, mode='bilinear')

  print("objs in imgs: {}".format(obj_to_img.shape))
  print("patches made: {}".format(img_patches.shape))
  return img_patches

def build_masks(imgs, objs, boxes, obj_to_img, f_inds):
  H, W = imgs.shape[2], imgs.shape[3]
  print("H: {}, W: {}".format(H,W))
  masks = torch.zeros((imgs.shape[0], 1, H, W))
  fore_objects = torch.cuda.LongTensor(f_inds['foreground_idx'])
  # print("F INDS: " + str(fore_objects))
  
  # use only boxes of hand picked foreground objects
  # boxes_inds = (objs in f_inds).nonzero().squeeze(1)
  s_inds = (obj_to_img == 3).nonzero().squeeze(1)
  print("OBJS: " + str(objs[s_inds]))
  boxes_inds = [i for i, obj in enumerate(objs) if obj in fore_objects]
  # print("BOXES INDS: " + str(boxes_inds))
  

  print("BOXES: " + str(boxes[s_inds]))
  # create masks
  for ind in boxes_inds:
    img_ind = obj_to_img[ind]
    if(img_ind == 2):
      print("box: " + str(boxes[ind]))
    # print("BOX BEFORE: " + str(boxes[ind]))
    # box = boxes[ind]
    # box = 

    box = torch.trunc(boxes[ind] * H).int()
    if(img_ind == 2):
      print("box after: " + str(box))
    masks[img_ind, :, box[1]:box[3] + 1, box[0]:box[2] + 1] = 1
  
  return masks

def main(args):
  if not os.path.isfile(args.checkpoint):
    print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
    print('Maybe you forgot to download pretraind models? Try running:')
    print('bash scripts/download_models.sh')
    return

  if not os.path.isdir(args.output_dir):
    print('Output directory "%s" does not exist; creating it' % args.output_dir)
    os.makedirs(args.output_dir)

  if args.device == 'cpu':
    device = torch.device('cpu')
  elif args.device == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      print('WARNING: CUDA not available; falling back to CPU')
      device = torch.device('cpu')

  # Load the model, with a bit of care in case there are no GPUs
  map_location = 'cpu' if device == torch.device('cpu') else None
  checkpoint = torch.load(args.checkpoint, map_location=map_location)
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.to(device)

  # Load the scene graphs
  with open(args.scene_graphs_json, 'r') as f:
    scene_graphs = json.load(f)

  #copy scene graphs
  # temp_scene = scene_graphs
  # _, _, obj_to_img = model.encode_scene_graphs(temp_scene)

  # print("test: " + str(obj_to_img.shape))
  # Run the model forward
  z_rels = scene_graphs[0]['relationships']
  
  for s, p, o in z_rels:
    print("P: " + str(p))

  with torch.no_grad():
    result = model.forward_json(scene_graphs)
    print(len(result[0]))
    imgs, boxes_pred, masks_pred, layouts = result[0][0], result[0][1], result[0][2]
    obj_to_img = result[1]
    triples = result[2]
    objs = result[3]

    i = 0
    for layout in layouts:
      imwrite('crop/layout-' + str(i) + '.png', layout)  
      i += 1
    # imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graphs)
    # check = model.forward_json(scene_graphs)
    z_inds = (obj_to_img == 0).nonzero().squeeze(1)
    print(triples.shape)
    print(objs.shape)
    print("MAH IMGS: " + str(imgs.shape))

  # print("shape: " + str((check)
  # print(len(check))


  imgs = imagenet_deprocess_batch(imgs, convert_range = False)

  x_box = boxes_pred[:,2] - boxes_pred[:,0]
  y_box = boxes_pred[:,3] - boxes_pred[:,1]
  # print(str(obj_to_img))
  # print(type(imgs))

  print("BOXES: " + str(boxes_pred.shape))
  print("objs: " + str(objs))

  fore_inds = json.load(open('datasets/vg/foreground.json', 'r'))
  masks = build_masks(imgs, objs, boxes_pred, obj_to_img, fore_inds)

  # patches = extract_patches(imgs, objs, boxes_pred, obj_to_img, fore_inds)

  cropped = crop_bbox_batch(imgs, boxes_pred, obj_to_img, 64)

  print(cropped.shape)

  i = 0
  for crop in cropped:
    img_crop = crop.numpy().transpose(1, 2, 0)
    imwrite('crop/' + str(obj_to_img.data.cpu().numpy()[i]) + " - " + str(i) + '.png', img_crop)
    i = i + 1

  # Save using torchvision
  save_image(imgs, "test.png", nrow=1, padding=0)

  # Save the generated images
  for i in range(imgs.shape[0]):
    test = imgs[i] * masks[i]
    test_path = os.path.join(args.output_dir, 'test%06d.png' % i)
    imwrite(test_path, test.numpy().transpose(1,2,0))

    img_np = imgs[i].numpy().transpose(1, 2, 0)
    img_path = os.path.join(args.output_dir, 'img%06d.png' % i)
    imwrite(img_path, img_np)


  # # Draw the scene graphs
  # if args.draw_scene_graphs == 1:
  #   for i, sg in enumerate(scene_graphs):
  #     sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
  #     sg_img_path = os.path.join(args.output_dir, 'sg{}.png'.format(i))
  #     imwrite(sg_img_path, sg_img)

  vocab = json.load(open('datasets/vg/vocab.json', 'r'))
  export_relationship_graph(vocab, triples[z_inds], objs)

def export_relationship_graph(vocab, triples, objs):
    obj_names = vocab['object_idx_to_name']
    pred_names = vocab['pred_idx_to_name']

    test = vocab['pred_name_to_idx']
    for s, p, o in triples:
      print("P: " + str(p))
      s_ind = objs[s]
      o_ind = objs[o]

      print("{} ---{}---> {}".format(obj_names[s_ind], pred_names[p + 1], obj_names[o_ind]))
    
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)