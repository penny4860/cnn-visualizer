# -*- coding: utf-8 -*-
# https://github.com/tensorflow/models/tree/master/slim
# Visualization of the filters of VGG16, via gradient ascent in input space.

from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from src.utils import recon
from src.vgg import Vgg16

np.set_printoptions(precision=5, linewidth=2000, suppress=True)


class ImgGenerateModel:
    
    def __init__(self, input_tensor, activation):
        self.loss_op = self._create_loss_op(activation)
        self.grads_op = self._create_gradient_op()
        
    def _create_loss_op(self, activation):
        return tf.reduce_mean(activation)

    def _create_gradient_op(self):
        grads_op = tf.gradients(self.loss_op, X)[0]
        grads_op = grads_op / tf.sqrt(tf.reduce_mean(tf.square(grads_op))) + tf.constant(1e-5)
        return grads_op


if __name__ == '__main__':
    filter_index = 0
    
    # 1. Input Tensor
    X = tf.placeholder(tf.float32, [None, 128, 128, 3])

    # 2. VGG network instance
    vggnet = Vgg16(X)

    # 3. Image Generator instance
    gen = ImgGenerateModel(vggnet.input, vggnet.conv5_1[:, :, :, filter_index])
    
    # 4. recon image
    image = recon(vggnet, gen)

    plt.imshow(image)
    plt.show()



