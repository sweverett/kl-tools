'''
This module contains classes and methods for computing analytic estimators
of KL related quantities, such as the galaxy inclination, shear (eventually), etc.
'''

import numpy as np
from scipy.stats import norm
from scipy.special import erf

from kl_tools.kross.tfr import estimate_vtf

def mu_sini(v_maj, sin_i, v_tf, sigma_vmaj, sigma_vcirc):
    '''
    The mean of the composite v_maj * v_tf gaussians
    '''

    num = sigma_vcirc**2 * (v_maj * sin_i) + (sigma_vmaj**2 * v_tf)
    den = sigma_vmaj**2 + (sigma_vcirc**2 * sin_i**2)

    return num / den

def var_sini(sigma_vmaj, sigma_vcirc, sin_i):
    '''
    The variance of the composite v_maj * v_tf gaussians
    '''

    num = sigma_vmaj**2 * sigma_vcirc**2
    den = sigma_vmaj**2 + (sigma_vcirc**2 * sin_i**2)

    return num / den

def sigma_sini(sigma_vmaj, sigma_vcirc, sin_i):
    '''
    The standard deviation of the composite v_maj * v_tf gaussians
    '''

    return np.sqrt(var_sini(sigma_vmaj, sigma_vcirc, sin_i))

def lognormal_base10(mean_log10, sigma_log10, size=None):
    '''
    Generate samples from a base-10 log-normal distribution.
    '''

    # convert mean and sigma from log10 to natural logarithm space
    mean_ln = mean_log10 * np.log(10)
    sigma_ln = sigma_log10 * np.log(10)
    
    # Generate samples from the natural log-normal distribution
    samples_ln = np.random.lognormal(mean=mean_ln, sigma=sigma_ln, size=size)

    return samples_ln

def gaussian_product_sini(v_maj, v_tf, sigma_vmaj, sigma_vcirc, sin_i):
    '''
    This method computes the extra term in the gaussian product due to
    completing the square in the exponent. This is the term that is
    only dependent on input parameters and not on the variable of
    integration. We use our own implementation of the gaussian dist
    here to avoid issues with 1 / sin(i)
    '''

    # modified as we multiply by sini / sini
    var = sigma_vmaj**2 + (sigma_vcirc * sin_i)**2

    # modified as we multiply by sini / sini
    norm = sin_i / np.sqrt(2 * np.pi * var)
    # norm = 1 / np.sqrt(2 * np.pi * var)

    # modified as we multiply by sini^2 / sini^2
    exp = np.exp(-0.5 * (v_maj - v_tf*sin_i)**2 / var)

    answer1 = norm * exp

    if sin_i == 0:
        sin_i = 1e-10
    var = (sigma_vmaj/sin_i)**2 + sigma_vcirc**2
    norm = 1 / np.sqrt(2 * np.pi * var)
    exp = np.exp(-0.5 * (v_maj/sin_i - v_tf)**2 / var)
    answer_2 = norm * exp

    if not np.isclose(answer1, answer_2):
        import ipdb; ipdb.set_trace()

    return norm * exp

def estimate_sini_from_vtf_single(
        vmaj,
        vtf,
        sigma_vmaj,
        sigma_vtf=0.05,
        method='argmax'
        ):

    sigma_vcirc = np.log(10) * sigma_vtf * vtf

    sin_i_vals = np.linspace(0.0001, 1.0, 1000)

    posterior = np.zeros_like(sin_i_vals)
    for i, sin_i in enumerate(sin_i_vals):

        mu = mu_sini(vmaj, sin_i, vtf, sigma_vmaj, sigma_vcirc)
        sig = sigma_sini(sigma_vmaj, sigma_vcirc, sin_i)

        gauss_product = gaussian_product_sini(
            vmaj, vtf, sigma_vmaj, sigma_vcirc, sin_i
        )
        prior = sin_i
        
        posterior[i] = prior * gauss_product * (1.+erf(mu/(np.sqrt(2)*sig)))/2.

    if method == 'argmax':
        # Find the root of the posterior
        map_sini = sin_i_vals[np.argmax(posterior)]
    else:
        raise ValueError('Method not recognized')

    return map_sini
