# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class ImgGenerateModel:
    
    def __init__(self, input_tensor, activation, i):
        self.loss_op = self._create_loss_op(activation[:,:,:,i])
        self.grads_op = self._create_gradient_op(input_tensor)
        
    def _create_loss_op(self, activation):
        return tf.reduce_mean(activation)

    def _create_gradient_op(self, X):
        grads_op = tf.gradients(self.loss_op, X)[0]
        grads_op = grads_op / tf.sqrt(tf.reduce_mean(tf.square(grads_op))) + tf.constant(1e-5)
        return grads_op


def recon(vggnet, img_generator, h, w, n_iter=20):
    
    image = _init_images(w=w, h=h, random_seed=111)
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

