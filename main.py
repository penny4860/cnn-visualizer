#-*- coding: utf-8 -*-

'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).

Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K


class Visualizer(object):
    """CNN model의 layer를 activation하는 image를 generation하는 class"""
    
    def __init__(self, cnn_model, layer_name):
        self.cnn_model = cnn_model
        self.layer_op = self._create_activation_op(layer_name)

    def _create_activation_op(self, layer_name):
        layer_dict = dict([(layer.name, layer) for layer in self.cnn_model.layers[1:]])
        return layer_dict[layer_name].output

    def _create_loss_op(self, filter_index):
        loss = K.mean(self.layer_op[:, :, :, filter_index])
        return loss
    
    def create_grad_op(self, filter_index):
        def normalize(x):
            # utility function to normalize a tensor by its L2 norm
            return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

        loss_op = self._create_loss_op(filter_index)
        grads_op = K.gradients(loss_op, self.cnn_model.input)[0]
        grads_op = normalize(grads_op)
        return grads_op, loss_op


class VisualizerRunner:
    
    def __init__(self, visualizer):
        self.vis = visualizer
    
    def run(self, w, h, n_filters=4, iterations=20):
        
        kept_filters = []
        for i in range(0, n_filters):
    
            start_time = time.time()
            img, loss = self._recon(w, h, i, iterations)
            end_time = time.time()
            
            if loss > 0:
                kept_filters.append((img, loss))
    
            print('Filter %d processed in %ds' % (i, end_time - start_time))
        print(len(kept_filters))
        
    def _recon(self, w, h, filter_index, iterations):
        
        # this function returns the loss and grads given the input picture
        grads, loss = self.vis.create_grad_op(filter_index)
        
        iterate_op = K.function([self.vis.cnn_model.input], [loss, grads])
        image = self._initialize_random_image(w, h)

        # we run gradient ascent for 20 steps
        for _ in range(iterations):
            loss_value, grads_value = iterate_op([image])
            
            image += grads_value * step
    
            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break
            
        image = self._deprocess_image(image[0])
        return image, loss_value

    def _initialize_random_image(self, w, h):
        # we start from a gray image with some random noise
        image = np.random.random((1, img_width, img_height, 3))
        image = (image - 0.5) * 20 + 128
        return image

    def _deprocess_image(self, x):
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


def draw_image(filters, img_width, img_height, n, margin = 5):
    
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    filters.sort(key=lambda x: x[1], reverse=True)
    filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))
    
    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
    
    # save the result to disk
    imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
    

if __name__ == '__main__':
    
    # parameters
    img_width = 128
    img_height = 128
    layer_name = 'block5_conv1'
    n_filters = 4
    step = 1.                       # step size for gradient ascent
    n = 2                           # we will stich the best 64 filters on a 8 x 8 grid.
    model = vgg16.VGG16(weights='imagenet', include_top=False)

    vis = Visualizer(model, layer_name)
    runner = VisualizerRunner(vis)
    kept_filters = runner.run(img_width, img_height, n_filters=4, iterations=2)



