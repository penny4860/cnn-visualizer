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

# util function to convert a tensor into a valid image


class Visualizer(object):
    
    def __init__(self, cnn_model, layer_name, filter_index):
        self.cnn_model = cnn_model
        self.loss_op = self._create_loss_op(layer_name, filter_index)
        self.grad_op = self._create_grad_op()

    def _create_loss_op(self, layer_name, filter_index):
        layer_dict = dict([(layer.name, layer) for layer in self.cnn_model.layers[1:]])
        layer_activation = layer_dict[layer_name].output
        loss = K.mean(layer_activation[:, :, :, filter_index])
        return loss
    
    def _create_grad_op(self):
        def normalize(x):
            # utility function to normalize a tensor by its L2 norm
            return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

        grads = K.gradients(self.loss_op, self.cnn_model.input)[0]
        grads = normalize(grads)
        return grads
    
    def recon(self, w, h, iterations=20):
        # this function returns the loss and grads given the input picture
        iterate_op = K.function([self.cnn_model.input], [self.loss_op, self.grad_op])
        image = self._initialize_random_image(w, h)

        # we run gradient ascent for 20 steps
        for _ in range(iterations):
            loss_value, grads_value = iterate_op([image])
            
            image += grads_value * step
    
            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break
    
        return image, loss_value

    def _initialize_random_image(self, w, h):
        # we start from a gray image with some random noise
        image = np.random.random((1, img_width, img_height, 3))
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
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
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

    # 1. build the VGG16 network with ImageNet weights
    model = vgg16.VGG16(weights='imagenet', include_top=False)
    kept_filters = []
    
    for i in range(0, n_filters):

        start_time = time.time()
        
        vis = Visualizer(model, layer_name, filter_index=i)
        img, loss = vis.recon((img_height, img_width), 2)
        
        end_time = time.time()
        
        if loss > 0:
            img = deprocess_image(img[0])
            kept_filters.append((img, loss))

        print('Filter %d processed in %ds' % (i, end_time - start_time))
    
    print(len(kept_filters))
    
    draw_image(kept_filters, img_width, img_height, n)


