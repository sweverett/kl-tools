import numpy as np
import os
import utils
import transformation as transform
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class VelocityModel(object):
    '''
    Default velocity model. Can subclass to
    define your own model
    '''

    name = 'default'
    _model_params = ['v0', 'vcirc', 'r0', 'rscale', 'sini',
                     'theta_int', 'g1', 'g2', 'r_unit', 'v_unit']

    def __init__(self, model_pars):
        if not isinstance(model_pars, dict):
            t = type(model_pars)
            raise TypeError(f'model_pars must be a dict, not a {t}!')

        self.pars = model_pars

        self._check_model_pars()

        return

    def _check_model_pars(self):
        mname = self.name

        # Make sure there are no undefined pars
        for name, val in self.pars.items():
            if val is None:
                raise ValueError(f'{param} must be set!')
            if name not in self._model_params:
                raise AttributeError(f'{name} is not a valid model ' +\
                                     f'parameter for {mname} velocity model!')

        # Make sure all req pars are present
        for par in self._model_params:
            if par not in self.pars:
                raise AttributeError(f'{par} must be in passed parameters ' +\
                                     f'to instantiate {mname} velocity model!')

        return

class VelocityMap(object):
    '''
    Base class for a velocity map
    '''

    def __init__(self, model_name, model_pars):
        self.model_name = model_name
        self.model = build_model(model_name, model_pars)

        return

class VelocityMap2D(VelocityMap):
    '''
    Extra features for a 2D map

    It is helpful to compute various things in different coordinate
    planes. The available ones are defined as follows:

    disk: Face-on view of the galactic disk, no inclination angle.
          This will be cylindrically symmetric for most models

    gal:  Galaxy major/minor axis frame with inclination angle same as
          source plane. Will now be ~ellipsoidal for sini!=0

    source: View from the lensing source plane, rotated version of gal
            plane with theta = theta_intrinsic

    obs:  Observed image plane. Sheared version of source plane
    '''

    _planes = ['obs', 'source', 'gal', 'disk']

    def __init__(self, model_name, model_pars):
        '''
        model_name: The name of the velocity to model.
        model_pars: A dict with key:val pairs for the model parameters.
                    Must be a registered velocity model.
        '''

        super(VelocityMap2D, self).__init__(model_name, model_pars)

        self._setup_transformations()

        return

    def _setup_transformations(self):
        pars = self.model.pars
        g1, g2 = pars['g1'], pars['g2']

        # Lensing transformation
        self.obs2source = np.array([
            [1-g1, -g2],
            [-g2, 1+g1]
        ])

        # Rotation by intrinsic angle
        theta_int = pars['theta_int']
        theta = -theta_int # want to 'subtract' orientation

        c, s = np.cos(theta), np.sin(theta)

        self.source2gal = np.array([
            [c, -s],
            [s,  c]
        ])

        # Account for inclination angle
        sini = pars['sini']
        i = np.arcsin(sini)
        self.gal2disk = np.array([
            [1, np.cos(i)],
            [0, 0]
        ])

        return

    def __call__(self, plane, x, y):
        '''
        Evaluate the velocity map at position (x,y) in the given plane. Note
        that position must be defined in the same plane
        '''

        if plane not in self._planes:
            raise ValueError(f'{plane} not a valid image plane!')

        if not isinstance(x, np.ndarray):
            assert not isinstance(y, np.ndarray)
            x = np.array(x)
            y = np.array(y)

        if x.shape != y.shape:
            raise Exception('x and y arrays must be the same shape!')

        return self._eval_map_in_plane(plane, x, y)

    def _eval_map_in_plane(self, plane, x, y):
        '''
        We use static methods defined in transformation.py
        to speed up these very common function calls

        The input (x,y) position is defined in the plane

        # The input (x,y) position is transformed from the obs
        # plane to the relevant plane
        '''

        pars = self.model.pars

        if plane == 'obs':
            func = transform._eval_in_obs_plane
        elif plane == 'source':
            func = transform._eval_in_source_plane
        elif plane == 'gal':
            func = transform._eval_in_gal_plane
        elif plane == 'disk':
            func = transform._eval_in_disk_plane

        return func(pars, x, y)

    def plot(self, plane, x=None, y=None, rmax=None, show=True, close=True,
             title=None, size=None, center=True, outfile=None):
        '''
        Plot velocity map in given plane. Will create a (x,y) grid
        based off of rmax centered at 0 in the chosen plane unless
        (x,y) are passed
        '''

        if plane not in self._planes:
            raise ValueError(f'{plane} not a valid image plane to plot!')

        pars = self.model.pars

        if rmax is None:
            rmax = 3. * pars['r0']

        if (x is None) or (y is None):
            if (x is not None) or (y is not None):
                raise ValueError('Can only pass both (x,y), not just one')

            # make square position grid in given plane

            Nr = 100
            dr = rmax / (Nr+1)
            r = np.arange(0, rmax+dr, dr)

            dx = (2*rmax) / (2*Nr+1)
            xarr = np.arange(-rmax, rmax+dx, dx)

            x, y = np.meshgrid(xarr, xarr)

        else:
            # parse given positions
            assert type(x) == type(y)
            if not isinstance(x, np.ndarray):
                x = np.array(x)
                y = np.array(y)

            if x.shape != y.shape:
                raise ValueError('x and y must have the same shape!')

        V = self(plane, x, y)

        runit = pars['r_unit']
        vunit = pars['v_unit']
        extent = [-rmax, rmax, -rmax, rmax]

        plt.imshow(V, origin='lower', extent=extent)
        plt.colorbar(label=f'{vunit}')

        runit = pars['r_unit']
        plt.xlabel(f'{runit}')
        plt.ylabel(f'{runit}')

        if center is True:
            plt.plot(0, 0, 'r', ms=10, markeredgewidth=2, marker='x')

        if title is not None:
            plt.title(title)
        else:
            r0 = pars['r0']
            plt.title(f'Velocity map in {plane} plane\n' +\
                      f'r_0 = {r0} {vunit}')

        if size is not None:
            plt.gcf().set_size_inches(size)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

    def plot_all_planes(self, plot_kwargs={}, show=True, close=True, outfile=None,
                        size=(9, 8)):
        if size not in plot_kwargs:
            plot_kwargs['size'] = size

        # Overwrite defaults if present
        if show in plot_kwargs:
            show = plot_kwargs['show']
            del plot_kwargs['show']
        if close in plot_kwargs:
            close = plot_kwargs['close']
            del plot_kwargs['close']
        if outfile in plot_kwargs:
            outfile = plot_kwargs['outfile']
            del plot_kwargs['outfile']

        pars = self.model.pars

        if 'rmax' in plot_kwargs:
            rmax = plot_kwargs['rmax']
        else:
            rmax = 3. * pars['r0']
            plot_kwargs['rmax'] = rmax

        # Start with square grid centered in disk plane
        Nr = 100
        dr = rmax / (Nr+1)
        r = np.arange(0, rmax+dr, dr)

        dx = (2*rmax) / (2*Nr+1)
        x = np.arange(-rmax, rmax+dx, dx)
        y = x

        Xdisk, Ydisk = np.meshgrid(x,y)

        # Figure out subplot grid
        Nplanes = len(self._planes)
        sqrt = np.sqrt(Nplanes)

        if sqrt % 1 == 0:
            N = sqrt
        else:
            N = int(sqrt+1)

        k = 1
        for plane in self._planes:
        # for plane in ['disk', 'gal']:
            plt.subplot(N,N,k)
            # pudb.set_trace()
            # X, Y = transform.transform_coords(Xdisk, Ydisk, 'disk', plane, pars)
            # X,Y = Xdisk, Ydisk
            X,Y = transform.transform_coords(Xdisk, Ydisk, 'disk', 'obs', pars)
            self.plot(plane, x=X, y=Y, show=False, close=False, **plot_kwargs)
            k += 1

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

def get_model_types():
    return MODEL_TYPES

# NOTE: This is where you must register a new model
MODEL_TYPES = {
    'default': VelocityModel,
    }

def build_model(name, pars, logger=None):
    name = name.lower()

    if name in MODEL_TYPES.keys():
        # User-defined input construction
        model = MODEL_TYPES[name](pars)
    else:
        raise ValueError(f'{name} is not a registered velocity model!')

    return model

def main(args):
    '''
    For now, just used for testing the classes
    '''

    show = args.show

    model_name = 'default'
    model_pars = {
        'v0': 200, # km/s
        'vcirc': 25, # km/s
        'r0': 10, # kpc
        'rscale': 10, #kpc
        'sini': 0.25,
        'theta_int': 1, # rad
        'r_unit': 'kpc',
        'v_unit': 'km/s',
        'g1': 0.02,
        'g2': 0.01
    }

    print('Creating VelocityMap2D from params')
    vmap = VelocityMap2D(model_name, model_pars)

    # for plane in ['disk', 'gal', 'source', 'obs']:
    #     print(f'Making plot of velocity field in {plane} plane')
    #     outfile = os.path.join(outdir, f'vmap-{plane}.png')
    #     vmap.plot(plane, outfile=outfile, show=show)

    print('Making combined velocity field plot')
    plot_kwargs = {}
    outfile = os.path.join(outdir, 'vmap-all.png')
    vmap.plot_all_planes(plot_kwargs=plot_kwargs, show=show, outfile=outfile)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test is True:
        outdir = os.path.join(utils.TEST_DIR, 'velocity')
        utils.make_dir(outdir)

        print('Starting tests')
        rc = main(args)

        if rc == 0:
            print('All tests ran succesfully')
        else:
            print(f'Tests failed with return code of {rc}')
