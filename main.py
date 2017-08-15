# -*- coding: utf-8 -*-
# https://github.com/tensorflow/models/tree/master/slim
# Visualization of the filters of VGG16, via gradient ascent in input space.

from __future__ import print_function
import tensorflow as tf

from src.utils import ImgGenerateModel, recon, plot_images
from src.vgg import Vgg16

if __name__ == '__main__':
    #################################################################################
    n_filters = 16
    n_iter = 20
    w = 64
    h = 64
    layer_name = 'conv1_2'
    #################################################################################
    
    # 1. Input Tensor
    X = tf.placeholder(tf.float32, [None, h, w, 3])

    # 2. VGG network instance
    vggnet = Vgg16(X)

    # 3. Image Generator instance
    images = []
    for i in range(n_filters):
        gen = ImgGenerateModel(vggnet.input, vggnet.get_activation(layer_name), i)
        image = recon(vggnet, gen, h, w, n_iter)
        images.append(image)

    plot_images(images)

