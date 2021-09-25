import types
import numpy as np
import os
from argparse import ArgumentParser
import zeus

import utils
from likelihood import log_posterior, PARS_ORDER

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class ZeusRunner(object):
    '''
    Class to run a MCMC chain with zeus

    Currently a very light wrapper around zeus, but in principle
    might want to do something fancier in the future
    '''

    def __init__(self, nwalkers, ndim, pfunc, args=None, kwargs=None):
        '''
        nwalkers: Number of MCMC walkers. Must be at least 2*ndim
        ndim:     Number of sampled dimensions
        pfunc:    Posterior function to sample from
        args:     List of additional args needed to evaluate posterior,
                    such as the data vector, covariance matrix, etc.
        kwargs:     List of additional kwargs needed to evaluate posterior,
                    such as the data vector, covariance matrix, etc.
        '''

        for name, val in {'nwalkers':nwalkers, 'ndim':ndim}.items():
            if val <= 0:
                raise ValueError(f'{name} must be positive!')
            if not isinstance(val, int):
                raise TypeError(f'{name} must be an int!')

        # Does not work for builtin functions, but that is fine here
        if not isinstance(pfunc, types.FunctionType):
            raise TypeError(f'{pfunc} is not a function!')

        if args is not None:
            if not isinstance(args, list):
                raise TypeError('args must be a list!')

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.pfunc = pfunc
        self.args = args
        self.kwargs = kwargs

        self._initialize_walkers()
        self._initialize_sampler()

        return

    def _initialize_walkers(self, scale=0.01):
        ''''
        TODO: Not obvious that this scale factor is reasonable
        for our problem, should experiment & test further

        Zeus reccommends to initialize in a small ball around the MAP
        estimate, but that is of course difficult to know a priori

        Might want to base this as some fractional scale for the width of
        each prior, centered at the max of the prior
        '''

        # self.start = np.zeros((self.nwalkers, self.ndim))
        self.start = 0.01*np.ones((self.nwalkers, self.ndim))

        # self.start = scale * np.random.randn(self.nwalkers, self.ndim)
        # self.start = scale * np.random.rand(self.nwalkers, self.ndim)

        # sini_indx = PARS_ORDER['sini']
        # self.start[:,sini_indx] *= np.sign(self.start[:,sini_indx])

        return

    def _initialize_sampler(self):
        self.sampler = zeus.EnsembleSampler(
            self.nwalkers, self.ndim, self.pfunc,
            args=self.args, kwargs=self.kwargs
            )

        return

    def run(self, nsteps, start=None, return_sampler=False, vb=True):
        '''
        nsteps: Number of MCMC steps / iterations
        start:  Can provide starting walker positions if you don't
                want to use the default initialization
        vb:     Will print out zeus summary if True
        return_sampler: Set to True if you want the sampler returned

        returns: zeus.EnsembleSampler object that contains the chains
        '''

        if start is None:
            self._initialize_walkers()
            start = self.start

        self.sampler.run_mcmc(start, nsteps)

        if vb is True:
            print(self.sampler.summary)

        if return_sampler is True:
            return self.sampler
        else:
            return

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'mcmc')
    utils.make_dir(outdir)

    print('Creating ZeusRunner object')
    ndims = 10
    nwalkers = 2*ndims
    args = None
    kwargs = None
    runner = ZeusRunner(nwalkers, ndims, log_posterior, args=args, kwargs=kwargs)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test is True:
        print('Starting tests')
        rc = main(args)

        if rc == 0:
            print('All tests ran succesfully')
        else:
            print(f'Tests failed with return code of {rc}')
