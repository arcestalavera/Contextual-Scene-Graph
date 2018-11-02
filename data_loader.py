import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
import json


class VGSceneLoader(object):
    def __init__(self, vocab, foreground_objs, imgs_path, h5_path, mode="train", image_size=(256, 256),
                 max_objects=10, include_relationships=True, use_orphaned_objects=True):
        """
        # Variables:
        # - h5_path: path of the h5 file of the preprocessed visual genome dataset
        """
        super(VGSceneLoader, self).__init__()
        self.max_objects = max_objects
        self.include_relationships = include_relationships
        self.use_orphaned_objects = use_orphaned_objects
        self.imgs_path = imgs_path

        # Transforms:
        # - Resize
        # - Convert to tensor
        # - Normalize using Imagenet mean and std dev

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

        # Load vocab
        self.vocab = vocab
        self.foreground_objs = foreground_objs

        # Dictionary for the data
        self.data = {}
        data_file = h5py.File(h5_path, 'r')
        for k, v in data_file.items():
            if k == 'image_paths':
                self.image_paths = list(v)
            else:
                self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        return self.data['object_names'].size(0)

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples:- LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        img_path = os.path.join(self.imgs_path, self.image_paths[index])
        
        # Open Image then transform
        image = Image.open(img_path)
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(
            range(self.data['objects_per_image'][index]))

        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'].data.numpy()[index, r_idx]
            o = self.data['relationship_objects'].data.numpy()[index, r_idx]
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        # sample objects that have relationship
        # check if sampling got a foreground object. If not, sample again
        error = ''
        while True:
            obj_idxs = list(obj_idxs_with_rels)
            obj_idxs_without_rels = list(obj_idxs_without_rels)
                
            if len(obj_idxs) > self.max_objects - 1:
                error = 'over'
                obj_idxs = random.sample(obj_idxs, self.max_objects)
                
            # if objects with relationships are less than the max objects, sample from objects w/o relationship
            if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
                error = 'under'

                num_to_add = self.max_objects - 1 - len(obj_idxs)
                num_to_add = min(num_to_add, len(obj_idxs_without_rels))
                obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        
            if(any(obj_idx in self.data['object_names'][index, obj_idxs] for obj_idx in self.foreground_objs)):
                    break
            else:
                print("Object List: {}".format(self.data['object_names'][index, list(obj_idxs_with_rels)]))
                print("resampling cause: {}. sampled: {}".format(error, self.data['object_names'][index, obj_idxs]))

        O = len(obj_idxs) + 1

        objs = torch.LongTensor(O).fill_(-1)
        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx]
            x, y, w, h = self.data['object_boxes'][index, obj_idx]
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i

        # The last object will be the special __image__ object
        objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        triples = []
        for r_idx in range(self.data['relationships_per_image'][index]):
            if not self.include_relationships:
                break
            s = self.data['relationship_subjects'].data.numpy()[index, r_idx]
            p = self.data['relationship_predicates'].data.numpy()[index, r_idx]
            o = self.data['relationship_objects'].data.numpy()[index, r_idx]
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        # Add dummy __in_image__ relationships for all objects
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)

        return image, objs, boxes, triples

def vg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """

    # batch is a list, and each element is (image, objs, boxes, triples)
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None])
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    # print("ALL IMGS: " + str(all_imgs.shape))
    all_objs = torch.cat(all_objs)
    # print("ALL OBJS: " + str(all_objs.shape))
    all_boxes = torch.cat(all_boxes)
    # print("ALL BOXES: " + str(all_boxes.shape))
    all_triples = torch.cat(all_triples)
    # print("ALL TRIPLES: " + str(all_triples.shape))
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples,
           all_obj_to_img, all_triple_to_img)
    return out

def vg_uncollate_fn(batch):
    """
    Inverse operation to the above.
    """
    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
    out = []
    obj_offset = 0
    for i in range(imgs.size(0)):
        cur_img = imgs[i]
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)
        cur_objs = objs[o_idxs]
        cur_boxes = boxes[o_idxs]
        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        obj_offset += cur_objs.size(0)
        out.append((cur_img, cur_objs, cur_boxes, cur_triples))
    return out

def get_loader(vocab, foreground_objs, imgs_path, h5_path, batch_size, image_size = (256, 256), mode = 'train'):
    """Build and return data loader."""

    # dataset = CelebALoader(data_path, attr_list, mode, img_size)
    dataset = VGSceneLoader(vocab, foreground_objs, imgs_path, h5_path, image_size = image_size)
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=vg_collate_fn)
    return data_loader

class CelebALoader(object):
    # AVAILABLE_ATTR = [
    #     "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    #     "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    #     "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    #     "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    #     "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    #     "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    #     "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    #     "Wearing_Necklace", "Wearing_Necktie", "Young"
    # ]

    AVAILABLE_ATTR = ["Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "bangs", "receding", "straight_hair",
                      "wavy", "hat", "bushy eyebrows", "eyeglasses", "mouth_slightly_open", "mustache", "smiling", "heavy_makeup", "pale", "Male"]
    hair_group = ["Black_Hair", "Blond_Hair",
                  "Brown_Hair", "Gray_Hair", "Bald"]

    def __init__(self, image_path, attributes_list, mode="train", img_size=96):
        """
        Initialize the data sampler with training data.
        """
        self.data = h5py.File(image_path, 'r')
        self.N = self.data[mode + "_images"].shape[0]
        self.mode = mode
        self.attributes_list = attributes_list

        if len(self.attributes_list) == 0:
            self.attr_idx = []
        else:
            self.attr_idx = self.get_attribute_index(self.attributes_list)

        self.transform = transforms.Compose([
            # transforms.Resize([img_size, img_size]),
            transforms.RandomCrop([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def get_attribute_index(self, names):
        # Check if names is a list
        if type(names) == type([]):
            output = []
            for name in names:
                output.append(self.AVAILABLE_ATTR.index(name))
            return output

        # If it's a single string
        return AVAILABLE_ATTR.index(names)

    def toggle_attribute_label(self, y_label, attribute, value=1):
        # Check if names is a list

        self.attributes_list
        self.attr_idx

        hair_attr_idx = []

        if attribute in self.hair_group and value == 1:
            for hair in self.hair_group:
                if hair in self.attributes_list:
                    h_idx = self.attributes_list.index(hair)
                    y_label[:, h_idx] = 0

        attr_idx = self.attributes_list.index(attribute)

        y_label[:, attr_idx] = value

        return y_label

        # If it's a single string
        return AVAILABLE_ATTR.index(names)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.N

    def __getitem__(self, index):
        tmpImg = self.data[self.mode + "_images"][index] * 255
        image = Image.fromarray(np.uint8(tmpImg))

        if len(self.attributes_list) == 0:
            attributes = np.float32(self.data[self.mode + "_feature"][index])
        else:
            attributes = np.float32(
                self.data[self.mode + "_feature"][index])[self.attr_idx]
        identity = np.int(self.data[self.mode + "_class"][index])
        return self.transform(image), attributes, identity


# def get_loader(data_path, img_size, attr_list, batch_size, mode='train'):
