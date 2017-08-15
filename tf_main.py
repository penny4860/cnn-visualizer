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


filter_index = 0
if __name__ == '__main__':
    
    # 1. Build graph
    X = tf.placeholder(tf.float32, [None, 128, 128, 3])
    vggnet = Vgg16(X)

    # 2. activation_op / loss_op / grads_op
    activation_op = vggnet.conv5_1
    loss_op = tf.reduce_mean(activation_op[:,:,:,filter_index])
    grads_op = tf.gradients(loss_op, X)[0]
    grads_op = grads_op / tf.sqrt(tf.reduce_mean(tf.square(grads_op))) + tf.constant(1e-5)

    # 3. session
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        vggnet.load_ckpt(sess)
        image = initialize_random_images(random_seed=111)
        for i in range(20):
            loss_value, grads_value = sess.run([loss_op, grads_op], feed_dict={X:image})
            image += grads_value
            print("Iter : {}, activation_score : {}".format(i, loss_value))

            
    image = deprocess_image(image[0])
    plt.imshow(image)
    plt.show()



