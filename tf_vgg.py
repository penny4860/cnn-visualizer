# -*- coding: utf-8 -*-
# https://github.com/tensorflow/models/tree/master/slim

from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

vgg = nets.vgg

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
predictions = vgg.vgg_16(images)
variables = slim.get_variables_to_restore(exclude=['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'])

for v in variables:
    print(v.name)
print("==================================")
restorer = tf.train.Saver(variables)

init_assign_op, init_feed_dict = slim.assign_from_checkpoint('ckpts/vgg_16.ckpt', variables)
with tf.Session() as sess:
    sess.run(init_assign_op, init_feed_dict)
    filter_ = sess.run(variables[0])
    value = filter_[:,:,:,0].reshape(-1,)
    print(value)

