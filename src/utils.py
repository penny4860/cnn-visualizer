# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def recon(vggnet, img_generator, n_iter=20):
    
    image = _init_images(random_seed=111)
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
    image = _deprocess_image(image[0])
    return image


def _init_images(n_images=1, w=128, h=128, random_seed=None):
    """create random noisy images
    
    # Args
        w : 
        h : 
        random_seed :
    
    # Returns
        images : array, shape of (n_images, w, h, 3)
    
    """
    np.random.seed(random_seed)
    
    images = np.random.random((n_images, w, h, 3))
    images = (images - 0.5) * 20 + 128
    return images


def _deprocess_image(x):
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


if __name__ == '__main__':
    pass

