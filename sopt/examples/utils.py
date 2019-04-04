#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import skimage, skimage.transform, skimage.data
from scipy.stats import circmean



def getSampleObj(npix=256, mod_range=1, phase_range=np.pi):
    """Creates a sample object using stock data from the skimage module.
    
    Parameters:
    npix - 
    Number of pixels in each axis of the object
    
    mod_range - 
    Maximum value of the modulus for the object pixels.
    
    phase_range - 
    Maximum value of the phase for the object pixels.
    """
    mod_img = skimage.img_as_float(skimage.data.camera())[::-1,::-1]
    phase_img = skimage.img_as_float(skimage.data.immunohistochemistry()[:,:,0])[::-1,::-1]
    mod = skimage.transform.resize(mod_img, [npix, npix], 
                                   mode='wrap', preserve_range=True) 
    phase = skimage.transform.resize(phase_img, [npix,npix],
                                     mode='wrap', preserve_range=True)
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * phase_range
    mod = (mod - np.min(mod)) / (np.max(mod) - np.min(mod)) * mod_range
    return mod * np.exp(1j * phase)

