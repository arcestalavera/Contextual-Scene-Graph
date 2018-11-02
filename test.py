import h5py
import numpy as np
import torch
import random
from PIL import Image
import json
import data_loader as loader
from cont_model import ContextualSceneModel as CSModel


# test_model = CSModel()

# data = {}
# data_file = h5py.File('vg/test.h5', 'r')
# for k, v in data_file.items():
# 	print("k: " + str(k))
# 	# print("v: " + str(v))	
# 	if k == 'image_paths':
# 		image_paths = list(v)
# 	else:
# 		data[k] = torch.IntTensor(np.asarray(v))
# 		# print("V: {}".format(data[k]))
# 		if k == 'relationship_subjects' or k == 'relationship_objects':
# 			print("here: " + str(data[k]))

def export_relationship_graph(vocab, objs, triples, obj_to_img):
    obj_names = vocab['object_idx_to_name']
    pred_names = vocab['pred_idx_to_name']

    # Do it per image
    with open('test.txt', 'w') as graph_file:
	    for i in range(max(obj_to_img) + 1):
	    	print("===================================", file = graph_file)
	    	print("image: {}".format(i), file = graph_file)
	    	img_inds = (obj_to_img == i).nonzero()
	    	img_triples = triples[img_inds].view(-1, 3)
	    	for s, p, o in img_triples:
		    	if(pred_names[p] == '__in_image__'):
		    		continue
		    	s_label = obj_names[objs[s]]
		    	p_label = pred_names[p]
		    	o_label = obj_names[objs[o]]
		    	print("{} --- {} ---> {}".format(s_label, p_label, o_label), file = graph_file)
	    	print("===================================", file = graph_file)

foreground_objs = json.load(open('vg/foreground.json', 'r'))
vocab = json.load(open('vg/vocab.json', 'r'))

data_loader = loader.get_loader(
    vocab, foreground_objs['foreground_idx'], 'vg/images', 'vg/test.h5', 2, mode='test')

batch = next(iter(data_loader))

# # print("len: {}".format(len(data_loader)))
# for i, batch in enumerate(data_loader):
imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
# print("OBJ OUT: " + str(objs))
export_relationship_graph(vocab, objs, triples, obj_to_img)

# z_inds = (obj_to_img == 0).nonzero().squeeze(1)
# print("OBJS: " + str(objs[z_inds]))
# print("BOXES: " + str(boxes[z_inds]))

# # input to stage 2 (refinement network)
# refine_in = coarse_out * mask + x * (1. - mask)
# refine_in = torch.cat([refine_in, ones, ones * mask], dim=1)
# refine_out, _ = self.refinement_net(refine_in, coarse_mask)

# x = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# x_inds = [1, 2]
# y_inds = [1, 2]

# test = [1, 2, 3, 4, 5, 6, 7]


# x[x_inds[0]:x_inds[1] + 1, y_inds[0]:y_inds[1] + 1] = 1

# print(x)

# f = h5py.File('vg/train.h5', 'r')
# data = {}
# for k, v in f.items():
#     print(str(k) + ": ")
#     print("V: " + str(list(v)[0]))
#     if(k == 'image_paths'):
#         image_paths = list(v)
#     else:
#         print("STORING K: " + str(k))
#         data[k] = torch.IntTensor(np.asarray(v))
# # img_path = os.path.join(self.image_dir, self.image_paths[index])
# image = Image.open('vg/images/' + str(image_paths[0]))
# WW, HH = image.size

# obj_idxs_with_rels = set()
# obj_idxs_without_rels = set(range(data['objects_per_image'][0]))
# print("OBJ W/O RELS: " + str(obj_idxs_without_rels))
# for r_idx in range(data['relationships_per_image'][0]):
# 	print("R_IDX: " + str(r_idx))
# 	s = data['relationship_subjects'].data.numpy()[0, r_idx]
# 	o = data['relationship_objects'].data.numpy()[0, r_idx]

# 	print("S: " + str(s))
# 	print("O: " + str(o))
# 	obj_idxs_with_rels.add(s)
# 	obj_idxs_with_rels.add(o)

# 	print("LEN OBJ WITH REL: " + str(len(list(obj_idxs_with_rels))))
# 	obj_idxs_without_rels.discard(s)
# 	obj_idxs_without_rels.discard(o)

# print("NEW OBJ W/O RELS: " + str(obj_idxs_without_rels))

# max_objects = 10
# obj_idxs = list(obj_idxs_with_rels)
# obj_idxs_without_rels = list(obj_idxs_without_rels)

# print("BEFORE: " + str(len(obj_idxs)))
# if len(obj_idxs) > max_objects - 1:
#     obj_idxs = random.sample(obj_idxs, max_objects)
# if len(obj_idxs) < max_objects - 1:
#     num_to_add = max_objects - 1 - len(obj_idxs)
#     num_to_add = min(num_to_add, len(obj_idxs_without_rels))
#     obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)

# print("AFTER: " + str(len(obj_idxs)))
# O = len(obj_idxs) + 1
# print("O: " + str(O))

# objs = torch.LongTensor(O).fill_(-1)
# print("OBJS: " + str(objs))

# boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
# print("BOXES: " + str(boxes))
# obj_idx_mapping = {}

# print("HH: " + str(HH))
# print("WW: " + str(WW))

# # Normalize boxes to range 0 - 1
# for i, obj_idx in enumerate(obj_idxs):
#     objs[i] = data['object_names'][0, obj_idx]
#     x, y, w, h = data['object_boxes'][0, obj_idx]
#     x0 = float(x) / WW
#     y0 = float(y) / HH
#     x1 = float(x + w) / WW
#     y1 = float(y + h) / HH

#     print(str(x) + " vs. " + str(x0))
#     print(str(y) + " vs. " + str(y0))
#     print(str(x + w) + " vs. " + str(x1))
#     print(str(y + h) + " vs. " + str(y1))
#     boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
#     obj_idx_mapping[obj_idx] = i

# print("obj idx: " + str(obj_idx))
# print("objs: " + str(objs))
# print("NEXT boxes: " + str(boxes))

# # The last object will be the special __image__ object
# objs[O - 1] = vocab['object_name_to_idx']['__image__']

# print("OBJS AFTER VOCAB: " + str(objs))

# # Create (s, p, o) triples for those objects with relationships
# triples = []
# for r_idx in range(data['relationships_per_image'][0]):
#     # if not self.include_relationships:
#     # break
#     s = data['relationship_subjects'].data.numpy()[0, r_idx]
#     p = data['relationship_predicates'].data.numpy()[0, r_idx]
#     o = data['relationship_objects'].data.numpy()[0, r_idx]
#     s = obj_idx_mapping.get(s, None)
#     o = obj_idx_mapping.get(o, None)
#     if s is not None and o is not None:
#         triples.append([s, p, o])
#     print("DIS MAH TRIPLES: " + str(triples))

# # Add dummy __in_image__ relationships for all objects
# in_image = vocab['pred_name_to_idx']['__in_image__']

# print("O SHOPPING: " + str(O))
# for i in range(O - 1):
#     triples.append([i, in_image, O - 1])

# triples = torch.LongTensor(triples)
# print("IM A TENSOR NOW TRIPLES: " + str(triples))
# return self.transform(image), boxes, triples
# if(k == 'relationship_subjects'):
# for i in range(rels):
# print("i: " + str(list(v)[i]))
# if(k == 'image_paths'):
# 	paths = list(v)

# 	for path in paths:
# 		if(path.startswith('VG_100K\\')):
# 			print(path)
# else:
# 	print(np.asarray(v))
