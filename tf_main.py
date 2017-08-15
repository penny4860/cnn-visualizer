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

def plot_images(images):
    
    n_images = len(images)
    n_rows = int(np.sqrt(n_images))
    n_cols = (n_images / n_rows) + 1
    
    fig, ax = plt.subplots()
    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(image)
        plt.axis("off")
    # plt.subplots_adjust(left=0, bottom=0, right=1.0, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()

if __name__ == '__main__':
    # 1. Input Tensor
    X = tf.placeholder(tf.float32, [None, 128, 128, 3])

    # 2. VGG network instance
    vggnet = Vgg16(X)

    # 3. Image Generator instance
    images = []
    for i in range(4):
        gen = ImgGenerateModel(vggnet.input, vggnet.conv5_1[:, :, :, i])
        image = recon(vggnet, gen)
        images.append(image)

    plot_images(images)

