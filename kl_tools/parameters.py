import pudb

'''
This file defines the structure and conversions between a params dict
and a parameter list (theta) that is used both in MCMC sampling and
in numba functions that won't accept a dict

pars: dict
    A dictionary that holds *at least* the defined params for the model

theta: list or np.array
    A list or numpy array that has the model pararmeters in a fixed order

TODO: To make this future-proof, what we should do is make a function that
      *returns* PARS_ORDER given a velocity model name. This way it is accessible
      throughout, but allows things to be a bit more flexible
'''

# order of sampled mcmc parameters
# NOTE: This won't be accessible if we use numba
PARS_ORDER = {
    'g1': 0,
    'g2': 1,
    'theta_int': 2,
    'sini': 3,
    'v0': 4,
    'vcirc': 5,
    'rscale': 6
    }

def theta2pars(theta):
    '''
    uses PARS_ORDER to convert list of sampled params to dict
    '''

    assert len(theta) == len(PARS_ORDER)

    pars = {}

    for key, indx in PARS_ORDER.items():
        pars[key] = theta[indx]

    return pars

def pars2theta(pars):
    '''
    convert dict of paramaeters to theta list
    '''

    # initialize w/ junk
    theta = len(PARS_ORDER) * ['']

    for name, indx in PARS_ORDER.items():
        theta[indx] = pars[name]

    return theta

