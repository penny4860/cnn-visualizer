# -*- coding: utf-8 -*-
# https://github.com/tensorflow/models/tree/master/slim
# Visualization of the filters of VGG16, via gradient ascent in input space.

from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np

from src.utils import initialize_random_images, deprocess_image

np.set_printoptions(precision=5, linewidth=2000, suppress=True)

class Vgg16(object):
    def __init__(self, input_tensor):
        # Build convolutional layers only
        self.conv1_1 = slim.conv2d(input_tensor, 64, [3, 3], scope='vgg_16/conv1/conv1_1')
        self.conv1_2 = slim.conv2d(self.conv1_1, 64, [3, 3], scope='vgg_16/conv1/conv1_2')
        self.pool1 = slim.max_pool2d(self.conv1_2, [2, 2], scope='pool1')
        
        self.conv2_1 = slim.conv2d(self.pool1, 128, [3, 3], scope='vgg_16/conv2/conv2_1')
        self.conv2_2 = slim.conv2d(self.conv2_1, 128, [3, 3], scope='vgg_16/conv2/conv2_2')
        self.pool2 = slim.max_pool2d(self.conv2_2, [2, 2], scope='pool2')
        
        self.conv3_1 = slim.conv2d(self.pool2, 256, [3, 3], scope='vgg_16/conv3/conv3_1')
        self.conv3_2 = slim.conv2d(self.conv3_1, 256, [3, 3], scope='vgg_16/conv3/conv3_2')
        self.conv3_3 = slim.conv2d(self.conv3_2, 256, [3, 3], scope='vgg_16/conv3/conv3_3')
        self.pool3 = slim.max_pool2d(self.conv3_3, [2, 2], scope='pool3')
         
        self.conv4_1 = slim.conv2d(self.pool3, 512, [3, 3], scope='vgg_16/conv4/conv4_1')
        self.conv4_2 = slim.conv2d(self.conv4_1, 512, [3, 3], scope='vgg_16/conv4/conv4_2')
        self.conv4_3 = slim.conv2d(self.conv4_2, 512, [3, 3], scope='vgg_16/conv4/conv4_3')
        self.pool4 = slim.max_pool2d(self.conv4_3, [2, 2], scope='pool4')
         
        self.conv5_1 = slim.conv2d(self.pool4, 512, [3, 3], scope='vgg_16/conv5/conv5_1')
        self.conv5_2 = slim.conv2d(self.conv4_1, 512, [3, 3], scope='vgg_16/conv5/conv5_2')
        self.conv5_3 = slim.conv2d(self.conv4_2, 512, [3, 3], scope='vgg_16/conv5/conv5_3')

    def load_ckpt(self, sess, ckpt='ckpts/vgg_16.ckpt'):
        variables = slim.get_variables(scope='vgg_16')
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(ckpt, variables)
        sess.run(init_assign_op, init_feed_dict)

filter_index = 0
if __name__ == '__main__':
    
    # 1. Build graph
    X = tf.placeholder(tf.float32, [None, 128, 128, 3])
    vgg = Vgg16(X)

    # 2. activation_op / loss_op / grads_op
    activation_op = vgg.conv5_1
    loss_op = tf.reduce_mean(activation_op[:,:,:,filter_index])
    grads_op = tf.gradients(loss_op, X)[0]
    grads_op = grads_op / tf.sqrt(tf.reduce_mean(tf.square(grads_op))) + tf.constant(1e-5)

    # 3. session
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        vgg.load_ckpt(sess)
        image = initialize_random_images(random_seed=111)
        for i in range(20):
            loss_value, grads_value = sess.run([loss_op, grads_op], feed_dict={X:image})
            image += grads_value
            print("Iter : {}, activation_score : {}".format(i, loss_value))

            
    image = deprocess_image(image[0])
    plt.imshow(image)
    plt.show()



