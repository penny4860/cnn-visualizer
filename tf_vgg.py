# -*- coding: utf-8 -*-
# https://github.com/tensorflow/models/tree/master/slim

from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def load_vgg_16():
    variables = slim.get_variables()
    for v in variables:
        print(v.name, v.get_shape())
    print("==================================")
    
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint('ckpts/vgg_16.ckpt', variables)
    with tf.Session() as sess:
        sess.run(init_assign_op, init_feed_dict)
        filter_ = sess.run(variables[0])
        value = filter_[:,:,:,0].reshape(-1,)
        print(value)
    return sess

def build_vgg16(input_tensor):
    net = slim.conv2d(input_tensor, 64, [3, 3], scope='vgg_16/conv1/conv1_1')
    net = slim.conv2d(net, 64, [3, 3], scope='vgg_16/conv1/conv1_2')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    
    net = slim.conv2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_1')
    net = slim.conv2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    
    net = slim.conv2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_1')
    net = slim.conv2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_2')
    net = slim.conv2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    
    net = slim.conv2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_1')
    net = slim.conv2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_2')
    net = slim.conv2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_3')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    
    net = slim.conv2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_1')
    net = slim.conv2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_2')
    net = slim.conv2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_3')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    build_vgg16(X)
    load_vgg_16()

 
