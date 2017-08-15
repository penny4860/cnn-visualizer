# -*- coding: utf-8 -*-
# https://github.com/tensorflow/models/tree/master/slim
# Visualization of the filters of VGG16, via gradient ascent in input space.

from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from src.utils import initialize_random_images, deprocess_image
from src.vgg import Vgg16

np.set_printoptions(precision=5, linewidth=2000, suppress=True)


class ImgGenerator:
    
    def __init__(self, input_tensor, activation):
        self.loss_op = self._create_loss_op(activation)
        self.grads_op = self._create_gradient_op()
        
    def _create_loss_op(self, activation):
        return tf.reduce_mean(activation)

    def _create_gradient_op(self):
        grads_op = tf.gradients(self.loss_op, X)[0]
        grads_op = grads_op / tf.sqrt(tf.reduce_mean(tf.square(grads_op))) + tf.constant(1e-5)
        return grads_op


def recon(vggnet, img_generator, n_iter=20):
    image = initialize_random_images(random_seed=111)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        vggnet.load_ckpt(sess)

        for i in range(n_iter):
            loss_value, grads_value = sess.run([img_generator.loss_op, img_generator.grads_op],
                                               feed_dict={vggnet.input:image})
            image += grads_value
            print("Iter : {}, activation_score : {}".format(i, loss_value))
            
    # image (1, w, h, 3)
    return deprocess_image(image[0])

if __name__ == '__main__':
    filter_index = 0
    
    # 1. Input Tensor
    X = tf.placeholder(tf.float32, [None, 128, 128, 3])

    # 2. VGG network instance
    vggnet = Vgg16(X)

    # 3. Image Generator instance
    gen = ImgGenerator(vggnet.input, vggnet.conv5_1[:, :, :, filter_index])
    
    # 4. recon image
    image = recon(vggnet, gen)

    plt.imshow(image)
    plt.show()



