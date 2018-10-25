from model import Generator
from torch.autograd import Variable
import torch
import tensorflow as tf
import numpy as np

x = Generator().cuda()
# print("here: " + str(x.refinement_net.refine_encoder))
dummy_x = Variable(torch.randn(3, 3, 256, 256).cuda())
dummy_mask = Variable(torch.randn(3, 1, 256, 256).cuda())

x(dummy_x, dummy_mask)

# a = [[[0, 1], [1,2]], [[1,1],[1,2]], [[0,1],[1,2]]]
# a_np = np.asarray(a, np.float32)

# a_tf = tf.convert_to_tensor(a_np, np.float32)
# b = tf.reduce_mean(a_tf, axis = [0], keep_dims=True)
# c = tf.reduce_mean(a_tf, axis = [0,1], keep_dims=True)
# d = tf.reduce_mean(a_tf, axis = [0,1,2], keep_dims=True)
# e = tf.reduce_mean(a_tf, axis = [1], keep_dims=True)
# test = a[0]
# with tf.Session() as sess:
# 	print(a_tf.eval())
# 	print("b " + str(b.eval()))
# 	print("c " + str(c.eval()))
# 	print("d " + str(d.eval()))
# 	print("e " + str(e.eval()))

# 	print("test " + str(a[0].eval()))