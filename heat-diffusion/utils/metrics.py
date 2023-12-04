import pandas
import jax.numpy as jnp
from utils import *

resolution = (32,32)
t1 = 0.2

from physics import forward_step

forward_step_fn = forward_step(resolution, t1) 

def get_mse(ground_truth, prediction):
    """
    Mean squared error between gt and prediction
    :param ground_truth:
    :param prediction:
    :return:
    """
    l2 = 0
    count = len(ground_truth)
    for i in range(count):
        l2 += jnp.mean((ground_truth[i] - (prediction[i] + forward_step_fn(prediction[i]))) ** 2)
        
    return l2 / count

def get_ps(elem):
    """
    Get the power spectrum
    :param elem:
    :return:
    """
    shape = elem.shape
    elem = np.abs(np.fft.fftshift(np.real(np.fft.ifftn(elem))))
    radial_profile_elem = np.log(radial_profile(elem, (int(shape[0]/2), int(shape[1]/2))))
    
    return radial_profile_elem
    

def get_ps_error(ground_truth, prediction):
    """
    Get the spectral error
    :param ground_truth: 
    :param prediction: 
    :return: 
    """
    s = int(jnp.ceil((resolution[0] / 2) * jnp.sqrt(2))) + 2

    weighting = 1
    cutoff = 10 + 1
    
    ps_error = 0
    count = len(ground_truth)
    for i in range(count):
        ps_gt = get_ps(ground_truth[i])
        ps_prediction = get_ps(prediction[i])
        ps_error += jnp.mean(1 * jnp.abs((ps_gt[1:cutoff] - ps_prediction[1:cutoff])))
        
    return ps_error / count