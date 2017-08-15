# -*- coding: utf-8 -*-
import numpy as np

def initialize_random_images(n_images=1, w=128, h=128, random_seed=None):
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


if __name__ == '__main__':
    image = initialize_random_images(random_seed=0)
    print(image.reshape(-1,)[0])
