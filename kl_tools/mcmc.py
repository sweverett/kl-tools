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
from corner import corner
import zeus
import emcee

import utils
import priors
from likelihood import DataCubeLikelihood
from velocity import VelocityMap

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class MCMCRunner(object):
    '''
    Class to run a MCMC chain with emcee or zeus

    Currently a very light wrapper around emzee/zeus, but in principle
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

        if not callable(pfunc):
            raise TypeError(f'{pfunc} is not callable!')

        if args is not None:
            if not isinstance(args, list):
                raise TypeError('args must be a list!')

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.pfunc = pfunc
        self.args = args
        self.kwargs = kwargs

        self.has_run = False
        self.has_MAP = False

        self.burn_in = None

        # Are set after a run with `compute_MAP()`
        self.MAP_means = None
        self.MAP_medians = None
        self.MAP_sigmas = None

        return

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

        if 'priors' in self.kwargs['pars']:
            # use peak of priors for initialization
            self.start = np.zeros((self.nwalkers, self.ndim))

            for name, indx in self.pars_order.items():
                prior = self.kwargs['pars']['priors'][name]
                peak, cen = prior.peak, prior.cen

                base = peak if peak is not None else cen
                radius = base*scale if base !=0 else scale

                # for (g1,g2), there seems to be some additional
                # outlier problems. So reduce
                if name in ['g1', 'g2']:
                    radius /= 2.

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

        outliers = np.abs(ball) > 2.*radius
        if isinstance(prior, priors.UniformPrior):
            left, right = prior.left, prior.right
            outliers = outliers | \
                        ((base + ball) < left) | \
                        ((base + ball) > right)
        elif isinstance(prior, priors.GaussPrior):
            if prior.clip_sigmas is not None:
                outliers = outliers | \
                    (abs(base + ball - prior.mu) > prior.clip_sigmas)
        Noutliers = len(np.where(outliers == True)[0])

        return outliers, Noutliers

    def run(self, nsteps, pool, start=None, return_sampler=False,
            vb=True):
        '''
        nsteps: int
            Number of MCMC steps / iterations
        pool: Pool
            A pool object returned from schwimmbad. Can be SerialPool,
            MultiPool, or MPIPool
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

        with pool:
            pt = type(pool)
            print(f'Pool: {pool}')

            if isinstance(pool, schwimmbad.MPIPool):
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)

            sampler = self._initialize_sampler(pool=pool)

            sampler.run_mcmc(
                start, nsteps, progress=progress
                )

        self.sampler = sampler

        self.has_run = True

        if return_sampler is True:
            return self.sampler
        else:
            return

    def set_burn_in(burn_in):
        self.burn_in = burn_in

        return

    def compute_MAP(self, discard=None, thin=1, recompute=False):
        '''
        TODO: For now, just computing the means & medians
              Will fail for multi-modal distributions

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

        chain = self.sampler.get_chain(flat=True, discard=discard, thin=thin)

        self.MAP_means   = np.mean(chain, axis=0)
        self.MAP_medians = np.median(chain, axis=0)

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
                    show=True, close=True, outfile=None, size=(16,16),
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

class KLensZeusRunner(ZeusRunner):
    '''
    Main difference is that we assume args=[datacube] and
    kwargs={pars:dict}
    '''

    def __init__(self, nwalkers, ndim, pfunc, datacube, pars):
        '''
        nwalkers: Number of MCMC walkers. Must be at least 2*ndim
        ndim:     Number of sampled dimensions
        pfunc:    Posterior function to sample from
        datacube: Datacube object the fit a model to
        pars: A Pars object containing the sampled pars and meta pars
              needed to evaluate posterior, such as
              covariance matrix, SED definition, etc.
        '''

        super(KLensZeusRunner, self).__init__(
            nwalkers, ndim, pfunc, args=[datacube], kwargs={
                'pars': pars.meta.pars
                }
            )

        self.datacube = datacube
        self.pars = pars

        self.pars_order = self.pars.sampled.pars_order

        self.MAP_vmap = None

        return

    def compute_MAP(self, discard=None, thin=1, recompute=False):
        super(KLensZeusRunner, self).compute_MAP(
            discard=discard, thin=thin, recompute=recompute
            )

        theta_pars = self.pars.theta2pars(self.MAP_medians)

        # Now compute the corresonding (median) MAP velocity map
        # vel_pars = theta_pars.copy()
        # vel_pars['r_unit'] = self.pars['r_unit']
        # vel_pars['v_unit'] = self.pars['v_unit']

        # self.MAP_vmap = VelocityMap('default', vel_pars)
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
                            size=(16,5)):
        '''
        For this subclass, have guaranteed access to datacube
        '''

        if self.has_MAP is False:
            print('MAP has not been computed yet; trying now ' +\
                  'with default parameters')
            self.compute_MAP()

        # gather needed components to evaluate model
        datacube = self.datacube
        lambdas = datacube.lambdas
        sed_array = DataCubeLikelihood._setup_sed(self.pars.meta.pars)
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
        # TODO: Eventually this should be called like vmap
        theta_pars = self.pars.theta2pars(self.MAP_medians)
        intensity = imap.render(theta_pars, datacube, self.pars.meta.pars)

        Nspec = datacube.Nspec

        fig, axs = plt.subplots(4, Nspec, sharex=True, sharey=True,)
        for i in range(Nspec):
            # first, data
            ax = axs[0,i]
            data = datacube.slices[i]._data
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
                lambdas[i], sed_array, zfactor, intensity
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

        return sampler

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
