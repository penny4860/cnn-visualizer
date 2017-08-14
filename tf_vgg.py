# -*- coding: utf-8 -*-
# https://github.com/tensorflow/models/tree/master/slim

from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt


def load_vgg_16(sess):
    variables = slim.get_variables()
    for v in variables:
        print(v.name, v.get_shape())
    print("==================================")
    
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint('ckpts/vgg_16.ckpt', variables)

    sess.run(init_assign_op, init_feed_dict)
    filter_ = sess.run(variables[0])
    value = filter_[:,:,:,0].reshape(-1,)
    print(value)

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
#     net = slim.conv2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_2')
#     net = slim.conv2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_3')
    return net

import numpy as np
def initialize_random_image(w=128, h=128):
    # we start from a gray image with some random noise
    image = np.random.random((1, w, h, 3))
    image = (image - 0.5) * 20 + 128
    return image

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x



filter_index = 0
if __name__ == '__main__':
    
    # 1. Build graph
    X = tf.placeholder(tf.float32, [None, 128, 128, 3])
    activation_op = build_vgg16(X)

    # 2. activation_op / loss_op / grads_op
    activation_op = activation_op
    loss_op = tf.reduce_mean(activation_op[:,:,:,filter_index])
    grads_op = tf.gradients(loss_op, X)[0]
    print(grads_op.get_shape())
    grads_op = grads_op / tf.sqrt(tf.reduce_mean(tf.square(grads_op))) + tf.constant(1e-5)
    print(grads_op.get_shape())

    # 3. session
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        load_vgg_16(sess)

        image = initialize_random_image()
        for _ in range(20):
            loss_value, grads_value = sess.run([loss_op, grads_op], feed_dict={X:image})
            image += grads_value
            print(loss_value)
            
    image = deprocess_image(image[0])
    plt.imshow(image)
    plt.show()



#     iterations = 10
#     for _ in range(iterations):
#         loss_value, grads_value = iterate_op([image])
#         
#         image += grads_value * step
# 
#         print('Current loss value:', loss_value)
#         if loss_value <= 0.:
#             # some filters get stuck to 0, we can skip them
#             break



