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

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def create_loss_tensor(layer_name):
    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])
    return loss

def create_grad_tensor(loss, variables):
    # we compute the gradient of the input picture wrt this loss
    # normalization trick: we normalize the gradient
    grads = K.gradients(loss, variables)[0]
    grads = normalize(grads)
    return grads

def random_gray_image():
    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        image = np.random.random((1, 3, img_width, img_height))
    else:
        image = np.random.random((1, img_width, img_height, 3))
    image = (image - 0.5) * 20 + 128
    return image

def gradient_ascent(loss_op, grads_op, input_img_pl, input_img_data):

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img_pl], [loss_op, grads_op])

    # we run gradient ascent for 20 steps
    for _ in range(4):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    return input_img_data, loss_value

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
    print('Model loaded.')
    model.summary()
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    kept_filters = []
    for filter_index in range(0, n_filters):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # create loss operation
        loss = create_loss_tensor(layer_name)
        grads = create_grad_tensor(loss, model.input)
        # input_img_data (1, 128, 128, 3)
        input_img_data, loss_value = gradient_ascent(loss, grads, model.input, random_gray_image())
    
        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
    
    draw_image(kept_filters, img_width, img_height, n)


