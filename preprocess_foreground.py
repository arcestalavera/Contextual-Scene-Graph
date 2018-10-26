import h5py
import torch
import os
import numpy as np

dataset = 'vg'
h5_path = os.path.join(dataset, 'train.h5')
foreground_path = os.path.join(dataset, 'foreground')

# Dictionary for the data
data = {}
data_file = h5py.File(h5_path, 'r')
for k, v in data_file.items():
	print("K: " + str(k))
	if k == 'image_paths':
		image_paths = list(v)
	else:
		data[k] = torch.IntTensor(np.asarray(v))

		print(v)