from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import truncnorm

# def log_prior(theta):
#     '''
#     theta: Parameters sampled by zeus. Order defined in PARS_ORDER
#     '''

#     # Until implemented, contribute nothing to posterior
#     return 0

class Prior(object):
    '''
    Each subclass should set a self.cen and self.peak attribute
    '''
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
        self.width = right - left
        self.inclusive = inclusive
        self.norm = 1. / (right - left)

        # There is no defined peak for a uniform dist
        self.peak = None
        self.cen = np.mean([left, right])
        self.scale = abs(right - left) / 4.
        # Boundary
        self.bound = [left, right]

        return

    def __call__(self, x, log=False, quantile=False):
        '''
        log: Set to return the log of the probability
        '''
        if quantile is True:
            return self._inv_cdf(x)
        
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

    def rvs(self, size=1):
        ''' Generate random sample from prior '''
        return np.random.uniform(low=self.left, high=self.right, size=size)
    
    def _inv_cdf(self, quantile):
        return self.left + self.width * quantile

class GaussPrior(Prior):
    def __init__(self, mu, sigma, clip_sigmas=None, zero_boundary=None):
        '''
        mu: mean of dist
        sigma: std of dist
        clip_sigmas: if set, reject samples beyond this many sigmas
                     from mu
        zero_boundary: str; ('positive', 'negative')
            Use to reject any samples above or below zero
        '''

        for p in [mu, sigma]:
            if not isinstance(p, (int, float)):
                raise TypeError('Prior parameters must be floats or ints!')

        if clip_sigmas is not None:
            if not isinstance(clip_sigmas, (int, float)):
                raise TypeError('clip_sigmas must be either an int or float!')
            if clip_sigmas <= 0:
                raise ValueError('clip_sigmas must be positive!')

        if zero_boundary is not None:
            if not isinstance(zero_boundary, str):
                raise TypeError('zero_boundary must be a str!')
            vals = ['positive', 'negative']
            if zero_boundary not in vals:
                raise ValueError(f'zero_boundary must be one of {vals}!')

        self.mu = mu
        self.sigma = sigma

        self.norm = 1. / (sigma * np.sqrt(2.*np.pi))
        self.clip_sigmas = clip_sigmas
        self.zero_boundary = zero_boundary

        self.peak = mu
        self.cen = mu
        self.scale = self.sigma
        # Boundary
        if self.clip_sigmas is not None:
            self.bound = [mu-clip_sigmas*sigma, mu+clip_sigmas*sigma]
        else: 
            self.bound = [-np.inf, np.inf]
        if zero_boundary=="positive":
            self.bound = [max(0, self.bound[0]), max(0, self.bound[1])]
        elif zero_boundary=="negative":
            self.bound = [min(0, self.bound[0]), min(0, self.bound[1])]
        return

    def __call__(self, x, log=False, quantile=False):
        '''
        log: Set to return the log of the probability
        '''
        if quantile is True:
            return self._inv_cdf(x)
        
        if self.clip_sigmas is not None:
            if (abs(x - self.mu) / self.sigma) > self.clip_sigmas:
                # sample clipped; ignore
                return -np.inf

        zb = self.zero_boundary
        if zb is not None:
            if zb == 'positive':
                if x < 0:
                    return -np.inf
            if zb == 'negative':
                if x > 0:
                    return -np.inf

        base = -0.5 * (x - self.mu)**2 / self.sigma**2

        if log is True:
            return np.log(self.norm) + base
        else:
            return norm * np.exp(base)

    def rvs(self, size=1):
        ''' Generate random sample from prior '''
        rand = truncnorm((self.bound[0]-self.mu)/self.sigma, 
                         (self.bound[1]-self.mu)/self.sigma, 
                         loc=self.mu, scale=self.sigma)
        return rand.rvs(size)
    
    def _inv_cdf(self, quantile):
        return scipy.stats.norm(self.mu, self.sigma).ppf(quantile)

class LognormalPrior(Prior):
    def __init__(self, mu, dex, clip_sigmas=None):
        '''
        mu: mean of dist (not log)
        dex: std in dex
        clip_sigmas: if set, reject samples beyond this many sigmas
                     from mu
        '''

        for p in [mu, dex]:
            if not isinstance(p, (int, float)):
                raise TypeError('Prior parameters must be floats or ints!')

        if clip_sigmas is not None:
            if not isinstance(clip_sigmas, (int, float)):
                raise TypeError('clip_sigmas must be either an int or float!')
            if clip_sigmas <= 0:
                raise ValueError('clip_sigmas must be positive!')

        self.mu = mu
        assert mu>0, f'mu must be positive in LognormalPrior!'
        self.dex = dex

        self.norm = 1. / (dex * np.sqrt(2.*np.pi))
        self.clip_sigmas = clip_sigmas

        self.peak = np.log10(mu)
        self.cen = np.log10(mu)
        self.scale = self.dex
        # Boundary
        if self.clip_sigmas is not None:
            self.bound = [self.cen-clip_sigmas*self.scale, self.cen+clip_sigmas*self.scale]
            self.bound = [10**self.bound[0], 10**self.bound[1]]
        else: 
            self.bound = [0, np.inf]
        return

    def __call__(self, x, log=False):
        '''
        log: Set to return the log of the probability
        '''
        if x<=0:
            return -np.inf
        if self.clip_sigmas is not None:
            if (abs(np.log10(x) - self.cen) / self.scale) > self.clip_sigmas:
                # sample clipped; ignore
                return -np.inf

        base = -0.5 * (np.log10(x) - self.cen)**2 / self.scale**2

        if log is True:
            return np.log(self.norm) + base
        else:
            return norm * np.exp(base)

    def rvs(self, size=1):
        ''' Generate random sample from prior '''
        rand = truncnorm((np.log10(self.bound[0])-self.cen)/self.scale, 
                         (np.log10(self.bound[1])-self.cen)/self.scale, 
                         loc=self.cen, scale=self.scale)
        return 10**(rand.rvs(size))
