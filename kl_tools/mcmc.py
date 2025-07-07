from abc import abstractmethod
import types
import numpy as np
import os
from multiprocessing import Pool
import schwimmbad
# from schwimmbad import SerialPool, MultiPool, MPIPool
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from corner import corner
import pickle
import zeus
import emcee
import pocomc as pc

import kl_tools.utils as utils
import kl_tools.priors as priors
from kl_tools.likelihood import DataCubeLikelihood
from kl_tools.velocity import VelocityMap

import ipdb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class MCMCRunner(object):
    '''
    Base class to run a MCMC chain (currently emcee, zeus, & pocomc)

    Currently a very light wrapper around a few samplers, but in principle
    might want to do something fancier in the future
    '''

    def __init__(self, nwalkers, ndim,
                 logpost=None, logpost_args=None, logpost_kwargs=None,
                 loglike=None, loglike_args=None, loglike_kwargs=None,
                 logprior=None, logprior_args=None, logprior_kwargs=None
                 ):
        '''
        nwalkers: int
            Number of MCMC walkers/particles. Must be at least 2*ndim for
            zeus/emcee, suggested to be at least 100 for pocomc
        ndim: int
            Number of sampled dimensions

        NOTE: The following are sets of callable functions & their args/kwargs
        needed to compute a (log) probablility density. The user should provide
        *either* a posterior or likelihood + prior, not both

        log{func}: function or callable()
            Callable function to sample from the log posterior, likelihood,
            or prior
        log{func}_args: list
            List of additional args needed to evaluate corresponding
            distribution, such as the data vector, covariance matrix, etc.
        log{func}_kwargs: dict (or subclass, such as Pars, MetaPars, etc.)
            Dictionary of additional kwargs needed to evaluate corresponding
            distribution, such as the meta parameters
        '''

        for name, val in {'nwalkers':nwalkers, 'ndim':ndim}.items():
            if val <= 0:
                raise ValueError(f'{name} must be positive!')
            if not isinstance(val, int):
                raise TypeError(f'{name} must be an int!')

        self.nwalkers = nwalkers
        self.ndim = ndim

        pfuncs = {}
        if (logpost is not None):
            if (loglike is not None) or (logprior is not None):
                raise Exception('Cannot provide both a posterior and ' +\
                                'a likelihood or prior!')
            self.logpost = logpost
            self.loglike = None
            self.logprior = None
            self.use_post = True
            pfuncs['posterior'] = logpost
        else:
            if (loglike is None) or (logprior is None):
                raise Exception('Must pass both a likelihood and prior ' +\
                                'if a posterior is not passed!')
            self.logpost = None
            self.loglike = loglike
            self.logprior = logprior
            self.use_post = False
            pfuncs['likelihood'] = loglike
            pfuncs['prior'] = logprior

        for name, pfunc in pfuncs.items():
            if not callable(pfunc):
                raise TypeError(f'passed log {name} is not callable!')
        self.pfuncs = pfuncs

        for arg in [logpost_args, loglike_args, logprior_args]:
            if arg is not None:
                if not isinstance(arg, list):
                    raise TypeError('passed args must be a list!')
        self.logpost_args = logpost_args
        self.loglike_args = loglike_args
        self.logprior_args = logprior_args

        for kwarg in [logpost_kwargs, loglike_kwargs, logprior_kwargs]:
            if kwarg is not None:
                if not isinstance(kwarg, dict):
                    raise TypeError('passed kwargs must be a dict!')
        self.logpost_kwargs = logpost_kwargs
        self.loglike_kwargs = loglike_kwargs
        self.logprior_kwargs = logprior_kwargs

        self.has_run = False
        self.has_MAP = False

        self.burn_in = None

        # Are set after a run with `compute_MAP()`
        self.MAP_means = None
        self.MAP_medians = None
        self.MAP_sigmas = None
        self.MAP_true = None # if actual loglikelihood values are passed
        self.MAP_indx = None # if actual loglikelihood values are passed

        self.sampler = None

        return

    @property
    def args(self):
        '''
        Most samplers use just a single args/kwargs pair. We define
        it here as the "default" one, even though some samplers
        have separate args for likelihood & prior
        '''
        pass

    @property
    def kwargs(self):
        '''
        Most samplers use just a single args/kwargs pair. We define
        it here as the "default" one, even though some samplers
        have separate kwargs for likelihood & prior
        '''
        pass

    @property
    def meta(self):
        pass

    @abstractmethod
    def _initialize_sampler(self, pool=None):
        pass

    def _initialize_walkers(self, scale=0.1):
        ''''
        TODO: Not obvious that this scale factor is reasonable
        for our problem, should experiment & test further

        Zeus reccommends to initialize in a small ball around the MAP
        estimate, but that is of course difficult to know a priori

        Might want to base this as some fractional scale for the width of
        each prior, centered at the max of the prior
        '''

        if 'priors' in self.meta:
            # use peak of priors for initialization
            self.start = np.zeros((self.nwalkers, self.ndim))

            for name, indx in self.pars_order.items():
                prior = self.meta['priors'][name]
                peak, cen = prior.peak, prior.cen

                base = peak if peak is not None else cen

                if prior.scale is not None:
                    radius = prior.scale
                elif base != 0:
                    radius = base*scale
                else:
                    radius = scale

                # random ball about base value
                ball = radius * np.random.randn(self.nwalkers)

                # rejcect 2+ sigma outliers or out of prior bounds
                outliers, Noutliers = self._compute_start_outliers(
                    base, ball, radius, prior
                    )

                # replace outliers
                while Noutliers > 0:
                    ball[outliers] = radius * np.random.randn(Noutliers)
                    outliers, Noutliers = self._compute_start_outliers(
                        base, ball, radius, prior
                        )

                self.start[:,indx] = base + ball

        else:
            # don't have much to go on
            self.start = scale * np.random.rand(self.nwalkers, self.ndim)

        return

    def _compute_start_outliers(self, base, ball, radius, prior):
        '''
        base: The reference value
        radius: The radius of the random ball
        ball: A ball of random points centered at 0 with given radius
        prior: prior being sampled with random points about ball
        '''

        outliers = np.abs(ball) > 3.*radius
        if isinstance(prior, priors.UniformPrior):
            left, right = prior.left, prior.right
            outliers = outliers | \
                        ((base + ball) <= left) | \
                        ((base + ball) >= right)
        elif isinstance(prior, priors.GaussPrior):
            if prior.clip_sigmas is not None:
                outliers = outliers | \
                    (abs(ball) > prior.clip_sigmas*prior.sigma)
        Noutliers = len(np.where(outliers == True)[0])

        return outliers, Noutliers

    def run(self, pool, nsteps=None, start=None, return_sampler=False,
            vb=True):
        '''
        pool: Pool
            A pool object returned from schwimmbad. Can be SerialPool,
            MultiPool, or MPIPool
        nsteps: int
            Number of MCMC steps / iterations
        start: list
            Can provide starting walker positions if you don't
            want to use the default initialization
        return_sampler: bool
            Set to True if you want the sampler returned
        vb: bool
            Will print out zeus summary if True

        returns: zeus.EnsembleSampler object that contains the chains
        '''

        if start is None:
            self._initialize_walkers()
            start = self.start

        if vb is True:
            progress = True
        else:
            progress = False

        if not isinstance(pool, schwimmbad.SerialPool):
            omp = int(os.environ['OMP_NUM_THREADS'])
            if omp != 1:
                print('WARNING: ENV variable OMP_NUM_THREADS is ' +\
                      f'set to {omp}. If not set to 1, this will ' +\
                      'degrade perfomance for parallel processing.')
        if pool is not None:
            with pool:
                pt = type(pool)
                #print(f'Pool: {pool}')

                if isinstance(pool, schwimmbad.MPIPool):
                    if not pool.is_master():
                        pool.wait()
                        sys.exit(0)

                self.sampler = self._initialize_sampler(pool=pool)

                self._run_sampler(start, nsteps=nsteps, progress=progress)
        else:
            self.sampler = self._initialize_sampler()

            self._run_sampler(start, nsteps=nsteps, progress=progress)

        self.has_run = True

        if return_sampler is True:
            return self.sampler
        else:
            return

    def _run_sampler(self, start, nsteps=None, progress=True):
        '''
        A standard way to run the sampler in the emcee/zeus convention.
        Can be overloaded by subclasses for different sampler calls
        '''

        if self.sampler is None:
            raise AttributeError('sampler has not yet been initialized!')

        if nsteps is None:
            raise Exception('nsteps should be set except for a few ' +\
                            'specific samplers!')

        self.sampler.run_mcmc(
            start, nsteps, progress=progress
            )

        return

    def set_burn_in(self, burn_in):
        self.burn_in = burn_in

        return

    def compute_MAP(self, loglike=None, discard=None, thin=1, recompute=False):
        '''
        loglike: np.ndarray, list
            A list or numpy array of log likelihood values from mcmc run

        # NOTE: the following are only used if loglike is not provided:
        discard: int
            The number of samples to discard, from 0:discard
        thin: int
            The factor by which to thin out the samples by; i.e.
            thin=2 will only use every-other sample
        '''

        if self.has_run is not True:
            print('Warning: Cannot compute MAP until the mcmc has been run!')
            return None

        if self.has_MAP is True:
            if recompute is False:
                raise ValueError('MAP values aready computed for this ' +
                                 'chain. To recompute with different ' +\
                                 'choices, use recompute=True')
            else:
                return

        if discard is None:
            if self.burn_in is not None:
                discard = self.burn_in
            else:
                raise ValueError('Must passs a value for discard if ' +\
                                 'burn_in is not set!')

        if loglike is None:
            # don't know actual min of loglikelihood, so do best we can
            chain = self.sampler.get_chain(
                flat=True, discard=discard, thin=thin
                )

            self.MAP_means   = np.mean(chain, axis=0)
            self.MAP_medians = np.median(chain, axis=0)

            self.MAP_sigmas = []
            for i in range(self.ndim):
                self.MAP_sigmas.append(np.percentile(chain[:, i], [16, 84]))
        else:
            chain = self.sampler.get_chain()
            self.MAP_indx = np.unravel_index(loglike.argmax(), loglike.shape)
            self.MAP_true = chain[self.MAP_indx]

            self.MAP_sigmas = []
            for i in range(self.ndim):
                self.MAP_sigmas.append(np.percentile(chain[:, i], [16, 84]))

        self.has_MAP = True

        return

    def plot_chains(self, burn_in=0, reference=None, show=True, close=True,
                    outfile=None, size=None):
        '''
        burn_in: int
            Set to discard the first 0:burn_in values of the chain
        reference: list
            Reference values to plot on chains, such as true values
        '''

        if self.has_run is not True:
            print('Warning: Cannot plot chains until mcmc has been run!')
            return

        chain = self.sampler.get_chain()

        ndim = self.ndim
        if size is None:
            size = (2*ndim, 1.25*ndim)

        fig = plt.figure(figsize=size)
        # for n in range(ndim):
        for name, indx in self.pars_order.items():
            plt.subplot2grid((ndim, 1), (indx, 0))
            plt.plot(chain[burn_in:,:,indx], alpha=0.5)
            plt.ylabel(name)

            if reference is not None:
                plt.axhline(reference[indx], lw=2, c='k', ls='--')

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

    def plot_corner(self, reference=None, discard=None, thin=1, crange=None,
                    show=True, close=True, outfile=None, size=(20,20),
                    show_titles=True, title=None, use_derived=True,
                    title_fmt='.3f'):
        '''
        reference: list
            Reference values to plot on chains, such as true or MAP values
        discard: int
            Set to throw out first 0:discard samples. Will use self.burn_in
            if already set
        thin: int
            Thins samples by the given factor
        crange: list
            A list of tuples or floats that define the parameter ranges or
            percentile fraction that is shown. Same as corner range arg
        use_derived: bool
            Turn on to plot derived parameters as well
        '''

        if self.has_run is not True:
            print('Warning: Cannot plot constraints until the mcmc has been run!')
            return

        if discard is None:
            if self.burn_in is not None:
                discard = self.burn_in
            else:
                raise ValueError('Must passs a value for discard if ' +\
                                 'burn_in is not set!')

        chain = self.sampler.get_chain(flat=True, discard=discard, thin=thin)

        if use_derived is True:
            # add derived quantity sini*vcirc
            new_shape = (chain.shape[0], chain.shape[1]+1)
            new_chain = np.zeros((new_shape))
            new_chain[:,0:-1] = chain
            i1, i2 = self.pars_order['sini'], self.pars_order['vcirc']
            new_chain[:,-1] = chain[:,i1] * chain[:,i2]
            chain = new_chain

        if reference is not None:
            if len(reference) != self.ndim:
                raise ValueError('Length of reference list must be same as Ndim!')

            if use_derived is True:
                ref = reference[i1]*reference[i2]
                if isinstance(reference, list):
                    reference.append(ref)
                elif isinstance(reference, np.ndarray):
                    arr = np.zeros(len(reference)+1)
                    arr[0:-1] = reference
                    arr[-1] = ref
                    reference = arr

            names = self.ndim*['']
            for name, indx in self.pars_order.items():
                names[indx] = name

            if use_derived is True:
                names.append('sini*vcirc')

        else:
            names = None

        if crange is not None:
            if use_derived is True:
                crange.append(crange[-1])
            if len(crange) != len(names):
                raise ValueError('Length of crange list must be same as names!')

        p = corner(
            chain, labels=names, truths=reference, range=crange,
            show_titles=show_titles, title_fmt=title_fmt
        )

        title_suffix = f'Burn in = {discard}'
        if title is None:
            title = title_suffix
        else:
            title += f'\n{title_suffix}'
        plt.suptitle(title, fontsize=18)

        if size is not None:
            plt.gcf().set_size_inches(size)

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

class ZeusRunner(MCMCRunner):

    def _initialize_sampler(self, pool=None):
        sampler = zeus.EnsembleSampler(
            self.nwalkers, self.ndim, self.pfunc,
            args=self.args, kwargs=self.kwargs, pool=pool
            )

        return sampler

    @property
    def args(self):
        return self.logpost_args

    @property
    def kwargs(self):
        return self.logpost_kwargs

    @property
    def pfunc(self):
        return self.logpost

    @property
    def meta(self):
        return self.pars.meta.pars

class KLensZeusRunner(ZeusRunner):
    '''
    Main difference is that we assume args=[datacube] and
    kwargs={pars:dict}
    '''

    def __init__(self, nwalkers, ndim, pfunc, datacube, pars):
        '''
        nwalkers: int
            Number of MCMC walkers. Must be at least 2*ndim
        ndim: int
            Number of sampled dimensions
        pfunc: function, callable()
            Posterior function to sample from
        datacube: DataCube
            A datacube object to fit a model to
        pars: A Pars object containing the sampled pars and meta pars
              needed to evaluate posterior, such as
              covariance matrix, SED definition, etc.
        '''

        super(KLensZeusRunner, self).__init__(
            nwalkers, ndim, logpost=pfunc, logpost_args=[datacube, pars],
            # nwalkers, ndim, logpost=pfunc, logpost_args=[datacube],
            # logpost_kwargs={'pars': pars.meta.pars}
            )

        self.datacube = datacube
        self.pars = pars

        self.pars_order = self.pars.sampled.pars_order

        self.MAP_vmap = None

        return

    def compute_MAP(self, loglike=None, discard=None, thin=1, recompute=False):
        super(KLensZeusRunner, self).compute_MAP(
            loglike=loglike, discard=discard, thin=thin, recompute=recompute
            )

        if self.MAP_true is None:
            theta_pars = self.pars.theta2pars(self.MAP_medians)
        else:
            theta_pars = self.pars.theta2pars(self.MAP_true)

        self.MAP_vmap = DataCubeLikelihood._setup_vmap(
            theta_pars, self.pars.meta.pars, 'default'
            )

        # Now do the same for the corresonding (median) MAP intensity map
        # TODO: For now, doing same simple thing in likelihood
        self.MAP_imap = DataCubeLikelihood._setup_imap(
            theta_pars, self.datacube, self.pars.meta
            )

        return

    def compare_MAP_to_truth(self, true_vmap, show=True, close=True,
                             outfile=None, size=(8,8)):
        '''
        true_vmap: VelocityMap object
            True velocity map
        '''

        Nx, Ny = self.datacube.Nx, self.datacube.Ny
        X, Y = utils.build_map_grid(Nx, Ny)

        Vtrue = true_vmap('obs', X, Y)
        Vmap  = self.MAP_vmap('obs', X, Y)

        fig, axes = plt.subplots(2,2, figsize=size, sharey=True)

        titles = ['Truth', 'MAP Model', 'Residual', '% Residual']
        images = [Vtrue, Vmap, Vmap-Vtrue, 100.*(Vmap-Vtrue)/Vtrue]
        for i in range(4):
            ax = axes[i//2, i%2]
            if '%' in titles[i]:
                vmin = np.max([-100., np.min(images[i])])
                vmax = np.min([ 100., np.max(images[i])])
            else:
                vmin, vmax = None, None
            im = ax.imshow(
                images[i], origin='lower', vmin=vmin, vmax=vmax
                )
            ax.set_title(titles[i])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

    def compare_MAP_to_data(self, show=True, close=True, outfile=None,
                            size=(24,5)):
        '''
        For this subclass, have guaranteed access to datacube
        '''

        if self.has_MAP is False:
            raise Exception('MAP has not been computed yet!')

        if self.MAP_true is None:
            theta_pars = self.pars.theta2pars(self.MAP_medians)
        else:
            theta_pars = self.pars.theta2pars(self.MAP_true)

        # gather needed components to evaluate model
        datacube = self.datacube
        lambdas = datacube.lambdas
        sed_array = DataCubeLikelihood._setup_sed(theta_pars, datacube)
        vmap = self.MAP_vmap
        imap = self.MAP_imap

        # create grid of pixel centers in image coords
        Nx = datacube.Nx
        Ny = datacube.Ny
        X, Y = utils.build_map_grid(Nx, Ny)

        # Compute zfactor from MAP velocity map
        V = vmap('obs', X, Y, normalized=True)
        zfactor = 1. / (1. + V)

        # compute intensity map from MAP
        intensity, continuum = imap.render(
            theta_pars, datacube, self.pars.meta.pars, im_type='both'
            )

        Nspec = datacube.Nspec

        # grab psf if present
        psf = datacube.get_psf()

        fig, axs = plt.subplots(4, Nspec, sharex=True, sharey=True,)
        for i in range(Nspec):
            # first, data
            ax = axs[0,i]
            data = datacube.data[i]
            im = ax.imshow(data, origin='lower')
            if i == 0:
                ax.set_ylabel('Data')
            l, r = lambdas[i]
            ax.set_title(f'({l:.1f}, {r:.1f}) nm')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

            # second, model
            ax = axs[1,i]
            model = DataCubeLikelihood._compute_slice_model(
                lambdas[i], sed_array, zfactor, intensity, continuum,
                psf=psf, pix_scale=datacube.pix_scale
                )
            im = ax.imshow(model, origin='lower')
            if i == 0:
                ax.set_ylabel('Model')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

            # third, residual
            ax = axs[2,i]
            residual = data - model
            im = ax.imshow(residual, origin='lower')
            if i == 0:
                ax.set_ylabel('Residual')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

            # fourth, % residual
            ax = axs[3,i]
            residual = 100. * (data - model) / model
            vmin = np.max([-100, np.min(residual)])
            vmax = np.min([ 100, np.max(residual)])
            im = ax.imshow(residual, origin='lower', vmin=vmin, vmax=vmax)
            if i == 0:
                ax.set_ylabel('% Residual')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

        if size is not None:
            plt.gcf().set_size_inches(size)
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

class KLensEmceeRunner(KLensZeusRunner):

    def _initialize_sampler(self, pool=None):
        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.pfunc,
            args=self.args, kwargs=self.kwargs, pool=pool
            )
        # test pickle dump to see the size of the posterior function
        # with open("test_JWST_post_func.pkl", 'wb') as f:
        #         pickle.dump(self.pfunc, f)
        return sampler

class PocoRunner(MCMCRunner):

    def _initialize_sampler(self, pool=None):
       sampler = pc.Sampler(
            self.nparticles, self.ndim,
            log_likelihood=self.loglike,
            log_prior=self.logprior,
            log_likelihood_args=self.loglike_args,
            log_likelihood_kwargs=self.loglike_kwargs,
            log_prior_args=self.logprior_args,
            log_prior_kwargs=self.logprior_kwargs,
            pool=pool,
            infer_vectorization=False
           # NOTE: wes bounds as we implement this in our priors
           #bounds=bounds
            )

        return sampler

class KLensPocoRunner(PocoRunner):
    '''
    See https://pocomc.readthedocs.io/en/latest/
    '''

    def __init__(self, nparticles, ndim, loglike, logprior,
                 datacube, pars,
                 loglike_args=None, loglike_kwargs=None,
                 logprior_args=None, logprior_kwargs=None):
        '''
        nparticles: int
            Number of MCMC particles. Recommended to be at least 100
            for complex posteriors
        ndim: int
            Number of sampled dimensions
        loglike: function / callable
            Log likelihood function to sample from
        logprior: function / callable
            Log prior function to sample from
        datacube: DataCube
            A datacube object to fit a model to
        pars: A Pars object containing the sampled pars and meta pars
              needed to evaluate posterior, such as
              covariance matrix, SED definition, etc.
        loglike_args: list
            List of additional args needed to evaluate log likelihood,
            such as the data vector, covariance matrix, etc.
        loglike_kwargs: dict
            List of additional kwargs needed to evaluate log likelihood,
            such as meta parameters, etc.
        logprior_args: list
            List of additional args needed to evaluate log prior,
            such as the data vector, covariance matrix, etc.
        logprior_kwargs: dict
            List of additional kwargs needed to evaluate log prior,
            such as meta parameters, etc.

        NOTE: to make this consistent w/ the other mcmc runner classes,
        you must pass datacube & pars separately from the rest of the
        args/kwargs!
        '''

       if loglike_args is not None:
           loglike_args = [datacube] + loglike_args
       else:
           loglike_args = [datacube]

       super(KLensPocoRunner, self).__init__(
           nparticles, ndim,
           loglike=loglike, logprior=logprior,
           loglike_args=loglike_args, loglike_kwargs=loglike_kwargs,
           logprior_args=logprior_args, logprior_kwargs=logprior_kwargs,
           )

        self.datacube = datacube
        self.pars = pars

        self.pars_order = self.pars.sampled.pars_order

        self.MAP_vmap = None

       #...

    @property
    def nparticles(self):
        return self.nwalkers

    @property
    def args(self):
        return self.loglike_args

    @property
    def kwargs(self):
        return self.loglike_kwargs

    @property
    def meta(self):
        return self.pars.meta.pars

    def _run_sampler(self, start, nsteps=None, progress=True):
        '''
        The poco-specific way to run the sampler object
        '''

        if self.sampler is None:
            raise AttributeError('sampler has not yet been initialized!')

        self.sampler.run(
            start, progress=progress
            )

        return

def get_runner_types():
    return RUNNER_TYPES

# NOTE: This is where you must register a new model
RUNNER_TYPES = {
    'default': None,
    'emcee': KLensEmceeRunner,
    'zeus': KLensZeusRunner,
    #'poco': KLensPocoRunner,
    }

def build_mcmc_runner(name, args, kwargs):
    '''
    name: str
        Name of mcmc runner type
    args: list
        A list of args for the runner constructor
    kwargs: dict, Pars, MCMCPars, etc.
        Keyword args to pass to runner constructor
    '''

    name = name.lower()

    if name in RUNNER_TYPES.keys():
        # User-defined input construction
        runner = RUNNER_TYPES[name](*args, **kwargs)
    else:
        raise ValueError(f'{name} is not a registered MCMC runner!')

    return runner

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'mcmc')
    utils.make_dir(outdir)

    print('Creating ZeusRunner object')
    ndims = 10
    nwalkers = 2*ndims
    args = None
    kwargs = None
    # runner = ZeusRunner(nwalkers, ndims, log_posterior, args=args, kwargs=kwargs)

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
