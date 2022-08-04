from copy import deepcopy
import utils

import ipdb

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

class Pars(object):
    '''
    Holds all of the parameters for a needed MCMC run, both
    sampled and meta parameters
    '''

    def __init__(self, sampled_pars, meta_pars):
        '''
        sampled_pars: list of str's
            A list of parameter names to be sampled in the MCMC.
            Their order will be used to define pars2theta
        meta_pars: dict
            A dictionary of meta parameters and their values for
            a particular experiment and MCMC run
        '''

        args = {
            'sampled_pars': (sampled_pars, list),
            'meta_pars': (meta_pars, dict)
            }
        utils.check_types(args)

        for name in sampled_pars:
            utils.check_type(name, 'sampled_par_val', str)

        pars_order = dict(zip(sampled_pars, range(len(sampled_pars))))
        self.sampled = SampledPars(pars_order)
        self.meta = MCMCPars(meta_pars)

        return

    def pars2theta(self, pars):
        return self.sampled.pars2theta(pars)

    def theta2pars(self, theta):
        return self.sampled.theta2pars(theta)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Pars(deepcopy(self.sampled), deepcopy(self.meta))

class SampledPars(object):
    '''
    Sets the structure for arbitrary sampled parameters, which
    are stored internally as a fixed list in the samplers
    '''

    def __init__(self, pars_order):
        '''
        pars_order: dict
            A dictionary that defines the par_name: sampler_index
            relationship in the used sampler

            For example:
            pars_order = {
                'g1': 0,
                'g2': 1,
                ...
            }
        '''

        utils.check_type(pars_order, 'pars_order', dict)

        for key, val in pars_order.items():
            if not isinstance(val, int):
                raise TypeError('pars_order must have int values!')

        self.pars_order = pars_order

        return

    def theta2pars(self, theta):
        '''
        uses pars_order to convert list of sampled params to dict
        '''

        assert len(theta) == len(self.pars_order)

        pars = {}
        for key, indx in self.pars_order.items():
            pars[key] = theta[indx]

        return pars

    def pars2theta(self, pars):
        '''
        convert dict of parameters to theta list
        '''

        assert len(pars) == len(self.pars_order)

        # initialize w/ junk that will fail if not set correctly
        theta = len(self.pars_order) * ['']

        for name, indx in self.pars_order.items():
            theta[indx] = pars[name]

        return theta

    def __repr__(self):
        return str(self.pars_order)

    def __len__(self):
        return len(self.pars_order.keys())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return SampledPars(deepcopy(self.pars_order))

class MetaPars(object):
    '''
    Base class that defines structure for the general meta
    parameters needed for an object, e.g. DataCube, likelihood
    MCMC, etc.
    '''

    _req_fields = []

    def __init__(self, pars):
        '''
        pars: dict
            Dictionary of meta pars
        '''

        utils.check_type(pars, 'pars', dict)

        self._check_pars(pars)
        self.pars = pars

        return

    @classmethod
    def _check_pars(cls, pars):
        '''
        Make sure that the general parameter list
        contains a few required entries
        '''

        for key in cls._req_fields:
            if key not in pars:
                return KeyError(f'{key} is a required field ' +\
                                'in the parameter list!')

        return

    def __getitem__(self, key):
        return self.pars[key]

    def __setitem__(self, key, val):
        self.pars[key] = val
        return

    def __delitem__(self, key):
        del self.pars[key]
        return

    def __iter__(self):
        return iter(self.pars)

    def __repr__(self):
        return str(self.pars)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return MetaPars(deepcopy(self.pars))

    def keys(self):
        return self.pars.keys()

    def items(self):
        return self.pars.items()

    def values(self):
        return self.pars.values()

class MCMCPars(MetaPars):

    '''
    Class that defines structure for the parameters
    used in MCMC sampling for a given experiment, modeling
    choices, etc.
    '''

    _req_fields = ['intensity', 'priors', 'units']
    _opt_fields = ['run_options', 'velocity', '_likelihood']

    def copy_with_sampled_pars(self, theta_pars):
        '''
        Scans through a MetaPars dict and sets any sampled meta pars
        to the current value in a returned copy

        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        '''

        pars = deepcopy(self.pars)

        return MCMCPars(self._set_sampled_pars(theta_pars, pars))

    @classmethod
    def _set_sampled_pars(cls, theta_pars, pars):
        '''
        Helper func for copy_with_sampled_pars()
        Assumes pars is already a copy of self.pars
        '''

        for key, val in pars.items():
            if isinstance(val, str) and (val.lower() == 'sampled'):
                pars[key] = theta_pars[key]

            elif isinstance(val, dict):
                pars[key] = cls._set_sampled_pars(theta_pars, pars[key])

        return pars

