# -*- coding: utf-8 -*-
# https://github.com/tensorflow/models/tree/master/slim
# Visualization of the filters of VGG16, via gradient ascent in input space.

from __future__ import print_function

import tensorflow as tf
import numpy as np

from src.utils import ImgGenerateModel, recon, plot_images
from src.vgg import Vgg16

np.set_printoptions(precision=5, linewidth=2000, suppress=True)

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

