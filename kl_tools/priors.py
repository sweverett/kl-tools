from abc import ABC, abstractmethod
import numpy as np

# def log_prior(theta):
#     '''
#     theta: Parameters sampled by zeus. Order defined in PARS_ORDER
#     '''

#     # Until implemented, contribute nothing to posterior
#     return 0

class Prior(object):
    @abstractmethod
    def __call__(self):
        pass

class UniformPrior(Prior):
    def __init__(self, left, right, inclusive=False):
        '''
        left: Left boundary for prior
        right: Right boundary for prior
        inclusive: Set to True to include bounds in prior range
        '''

        for b in [left, right]:
            if not isinstance(b, (int, float)):
                raise TypeError(f'Bounds must be ints or floats!')

        if left >= right:
            raise ValueError('left must be less than right!')

        if not isinstance(inclusive, bool):
            raise TypeError('inclusive must be a bool!')
        self.left = left
        self.right = right
        self.inclusive = inclusive
        self.norm = 1. / (right - left)

        return

    def __call__(self, x, log=False):
        '''
        log: Set to return the log of the probability
        '''

        val = self.norm

        if self.inclusive is True:
            if (x < self.left) or (x > self.right):
                val = 0
        else:
            if (x <= self.left) or (x >= self.right):
                val = 0

        if log is False:
            return val
        else:
            if val == 0:
                return -np.inf
            else:
                return np.log(val)

class GaussPrior(Prior):
    def __init__(self, mu, sigma):
        for p in [mu, sigma]:
            if not isinstance(p, (int, float)):
                raise TypeError('Prior parameters must be floats or ints!')

        self.mu = mu
        self.sigma = sigma
        self.norm = 1. / (sigma * np.sqrt(2.*np.pi))

        return

    def __call__(self, x, log=False):
        '''
        log: Set to return the log of the probability
        '''

        base = -0.5 * (x - self.mu)**2 / self.sigma**2

        if log is True:
            return np.log(self.norm) + base
        else:
            return norm * np.exp(base)
