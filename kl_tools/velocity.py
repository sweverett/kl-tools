import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as units

import kl_tools.transformation as transform
import kl_tools.numba_transformation as numba_transform
from kl_tools.utils import MidpointNormalize, plot
from kl_tools.plotting import plot_line_on_image
from kl_tools.transformation import TransformableImage
from kl_tools.parameters import SampledPars
from kl_tools.coordinates import OrientedAngle

class VelocityModel(object):
    '''
    Default velocity model. Can subclass to
    define your own model
    '''

    name = 'centered'
    _model_params = [
        'v0',
        'vcirc',
        'rscale',
        'sini',
        'theta_int',
        'g1',
        'g2',
        'r_unit',
        'v_unit'
        ]

    _ignore_params = ['beta', 'flux']

    def __init__(self, model_pars):
        if not isinstance(model_pars, dict):
            t = type(model_pars)
            raise TypeError(f'model_pars must be a dict, not a {t}!')

        self.pars = model_pars

        self._check_model_pars()

        # Needed if using numba for transformations
        # can set during MCMC w/ self.build_pars_array()
        self.pars_array = None

        # some models can include an offset, which adds a
        # transformation layer
        self.has_offset = False

        return

    def _check_model_pars(self):
        mname = self.name

        # Make sure there are no undefined pars
        for name, val in self.pars.items():
            if val is None:
                raise ValueError(f'{name} must be set!')
            if name not in self._model_params:
                if name in self._ignore_params:
                    continue
                else:
                    #TODO: figure out how to handle this more elegantly...
                    # now that we can marginalize over arbitrary pars,
                    # can't simply raise an error here
                    pass
                    # raise AttributeError(f'{name} is not a valid model ' +\
                    #                      f'parameter for {mname} velocity model!')

        # Make sure all req pars are present
        for par in self._model_params:
            if par not in self.pars:
                raise AttributeError(f'{par} must be in passed parameters ' +\
                                     f'to instantiate {mname} velocity model!')

        # Make sure units are astropy.unit objects
        unit_pars = {
            'r_unit': self.pars['r_unit'],
            'v_unit': self.pars['v_unit']
        }
        for name, u in unit_pars.items():
            if (not isinstance(u, units.Unit)) and \
               (not isinstance(u, units.CompositeUnit)) and \
               (not isinstance(u, units.IrreducibleUnit)):
               # try to convert to unit object
               self.pars[name] = units.Unit(u)

        return

    def build_pars_array(self, pars):
        '''
        Numba requires a pars array instead of a more flexible dict

        pars: Pars
            A pars object that can convert between a pars dict and
            sampled theta array
        '''

        sampled = SampledPars(self.pars)

        self.pars_arr = pars.pars2theta(self.pars)

        return

    def get_transform_pars(self):
        pars = {}
        for name in ['g1', 'g2', 'sini', 'theta_int']:
            pars[name] = self.pars[name]

        return pars

class OffsetVelocityModel(VelocityModel):
    '''
    Same as default velocity model, but w/ a 2D centroid offset
    '''

    name = 'offset'
    _model_params = [
        'v0',
        'vcirc',
        'rscale',
        'sini',
        'x0',
        'y0',
        'theta_int',
        'g1',
        'g2',
        'r_unit',
        'v_unit'
        ]

    _ignore_params = ['beta', 'flux']

    def __init__(self, model_pars):
        super(OffsetVelocityModel, self).__init__(model_pars)

        self.has_offset = True

        return

    def get_transform_pars(self):
        pars = {}
        for name in ['g1', 'g2', 'sini', 'theta_int', 'x0', 'y0']:
            pars[name] = self.pars[name]

        return pars

class VelocityMap(TransformableImage):
    '''
    Base class for a velocity map

    It is helpful to compute various things in different coordinate
    planes. The available ones are defined as follows:

    disk: Face-on view of the galactic disk, no inclination angle.
          This will be cylindrically symmetric for most models

    gal:  Galaxy major/minor axis frame with inclination angle same as
          source plane. Will now be ~ellipsoidal for sini!=0

    source: View from the lensing source plane, rotated version of gal
            plane with theta = theta_intrinsic

    cen: View from the object-centered observed plane. Sheared version of
         source plane

    obs:  Observed image plane. Offset version of cen plane
    '''

    def __init__(self, model_name, model_pars):
        '''
        model_name: str
            The name of the velocity to model.
        model_pars: dict
            A dict with key:val pairs for the model parameters.
            Must be a registered velocity model.
        '''

        self.model_name = model_name
        self.model = build_model(model_name, model_pars)

        transform_pars = self.model.get_transform_pars()
        super(VelocityMap, self).__init__(transform_pars)

        return

    def __call__(self, plane, x, y, speed=False, normalized=False,
                 use_numba=False, indexing='ij', pix_scale=None):
        '''
        Evaluate the velocity map at position (x,y) in the given plane. Note
        that position must be defined in the same plane

        Parameters
        ----------
        plane: str
            The plane to evaluate the velocity map in
        x, y: np.ndarray
            The position coordinates in the given plane. If 2D, see `indexing`
            arg for more info on allowed indexing conventions for these arrays
        speed: bool
            Set to True to return speed map instead of velocity
        normalized: bool
            Set to True to return velocity / c
        use_numba: bool
            Set to True to use numba versions of transformations
        # TODO: Clean this up when ready
        indexing: str
            If passing 2D numpy arrays for x & y, set the indexing to be either
            'ij' or 'xy'. Consistent with numpy.meshgrid `indexing` arg:
              - 'ij' if using matrix indexing; [i,j]=(x,y)
              - 'xy' if using cartesian indexing; [i,j]=(y,x) 
            If the equivalent 1D arrays are such that len(x) = M and len(y) = N,
            then the 2D arrays should be of shape (N,M) if indexing='xy' and
            (M,N) if indexing='ij'. Default is 'ij'. For more info, see:
            https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        pix_scale: float; optional
            The pixel scale of the grid. Required if r_unit is 'arcsec'
        '''

        if x.shape != y.shape:
            raise ValueError('x and y must have the same shape!')

        if len(x.shape) > 2:
            raise ValueError('x and y must be 1D or 2D arrays!')

        # if the arrays are 2D and using cartesian indexing, swap axes
        # to make them consistent with the transformation matrices defined
        # in transformation.py
        # TODO: Clean this up when ready
        if (len(x.shape) == 2) and (indexing == 'xy'):
            x = np.swapaxes(x, 0, 1)
            y = np.swapaxes(y, 0, 1)

        if self.model.pars['r_unit'] == units.arcsec:
            if pix_scale is None:
                import ipdb; ipdb.set_trace()
                raise ValueError('Must pass pix_scale if r_unit is arcsec!')
            self._tmp_pix_scale = pix_scale
        elif self.model.pars['r_unit'] == units.pixel:
            # ok to assume 1 if in pixel units
            self._tmp_pix_scale = 1
        else:
            raise ValueError('r_unit must be either arcsec or pixels!')

        super(VelocityMap, self).__call__(
            plane, x, y, use_numba=use_numba
            )

        if normalized is True:
            norm = 1. / const.c.to(self.model.pars['v_unit']).value
        else:
            norm = 1.

        if self.model.has_offset is True:
            offset = True
        else:
            offset = False

        rendered_vmap = norm * self._eval_map_in_plane(
            plane, offset, x, y, speed=speed, use_numba=use_numba
            )

        # reset the pixel scale for the next call
        self._tmp_pix_scale = None

        return rendered_vmap

    def _eval_map_in_plane(self, plane, offset, x, y, speed=False,
                           use_numba=False):
        '''
        We use static methods defined in transformation.py
        to speed up these very common function calls

        The input (x,y) position is defined in the plane

        The input (x,y) position is transformed from the obs
        plane to the relevant plane

        offset determines whether to include the centroid translation
        layer. Will skip if not needed

        speed: bool
            Set to True to return speed map instead of velocity
        use_numba: bool
            Set to True to use numba versions of transformations
        '''

        # Need to use array for numba
        if use_numba is True:
            pars = self.model.pars_arr
        else:
            pars = self.model.pars

        # needed for disk plane evaluation
        pars['pix_scale'] = self._tmp_pix_scale

        func = self._get_plane_eval_func(plane, use_numba=use_numba)

        return func(pars, x, y, speed=speed, offset=offset)

    @classmethod
    def _eval_in_obs_plane(cls, pars, x, y, **kwargs):
        '''
        pars: dict
            Holds the model & transformation parameters
        x,y: np.ndarray
            The position coordintates in the obs plane

        kwargs holds any additional params that might be needed
        in subclass evaluations, such as using speed instead of
        velocity
        '''

        cen_offset = kwargs['offset']

        # first evaluate vmap in obs plane without any systematic
        # velocity offset (distinct from centroid offset)
        if cen_offset is True:
            obs_vmap = super(VelocityMap, cls)._eval_in_obs_plane(
                pars, x, y, **kwargs
                )
        else:
            # skip the centroid offset translation layer
            obs_vmap = super(VelocityMap, cls)._eval_in_cen_plane(
                pars, x, y, **kwargs
                )

        # now add systematic velocity
        return pars['v0'] + obs_vmap

    @classmethod
    def _eval_in_gal_plane(cls, pars, x, y, **kwargs):
        '''
        pars: dict
            Holds the model & transformation parameters
        x,y: np.ndarray
            The position coordintates in the gal plane

        kwargs holds any additional params that might be needed
        in subclass evaluations, such as using speed instead of
        velocity
        '''

        xp, yp = transform._gal2disk(pars, x, y)

        # Need the speed map in either case
        speed = kwargs['speed']
        kwargs['speed'] = True
        speed_map = cls._eval_in_disk_plane(pars, xp, yp, **kwargs)

        # speed will be in kwargs
        if speed is True:
            return speed_map

        else:
            # euler angles which handle the vector aspect of velocity transform
            sini = pars['sini']
            phi = np.arctan2(yp, xp)

            return sini * np.cos(phi) * speed_map

    @classmethod
    def _eval_in_disk_plane(cls, pars, x, y, **kwargs):
        '''
        Evaluates model at positon array in the disk plane, where
        pos=(x,y) is defined relative to galaxy center

        pars is a dict with model parameters

        will eval speed map instead of velocity if speed is True
        '''

        if kwargs['speed'] is False:
            return np.zeros(np.shape(x))

        r = np.sqrt(x**2 + y**2)

        # NOTE / TODO: we should make the grid be more general and allow
        # it to be either in pixels or physical units. For now, we assume
        # that the grid is in pixel units
        r_unit = pars['r_unit']
        rscale = pars['rscale']
        if r_unit == 'arcsec':
            pix_scale = pars['pix_scale']
            rscale = (rscale /  pix_scale)
            if isinstance(rscale, units.Quantity):
                rscale = rscale.value

        atan_r = np.arctan(r  / rscale)

        v_r = (2./ np.pi) * pars['vcirc'] * atan_r

        return v_r

    def plot(self, plane, x=None, y=None, rmax=None, show=True, close=True,
             title=None, size=(9,8), center=True, outfile=None, speed=False,
             normalized=False, pix_scale=None):
        '''
        Plot speed or velocity map in given plane. Will create a (x,y)
        grid based off of rmax centered at 0 in the chosen plane unless
        (x,y) are passed

        Can normalize to v/c if desired
        '''

        if plane not in self._planes:
            raise ValueError(f'{plane} not a valid image plane to plot!')

        if (pix_scale is None) and (self.model.pars['r_unit'] == 'arcsec'):
            raise ValueError('Must pass pix_scale if r_unit is arcsec!')

        pars = self.model.pars

        if rmax is None:
            rmax = 5. * pars['rscale']

        if (x is None) or (y is None):
            if (x is not None) or (y is not None):
                raise ValueError('Can only pass both (x,y), not just one')

            # make square position grid in given plane
            Nr = 100
            dr = rmax / (Nr+1)
            r = np.arange(0, rmax+dr, dr)

            dx = (2*rmax) / (2*Nr+1)
            r = np.arange(-rmax, rmax+dx, dx)

            x, y = np.meshgrid(r, r, indexing='ij')

        else:
            # parse given positions
            assert type(x) == type(y)
            if not isinstance(x, np.ndarray):
                x = np.array(x)
                y = np.array(y)

            if x.shape != y.shape:
                raise ValueError('x and y must have the same shape!')

        V = self(
            plane, x, y, speed=speed, normalized=normalized, pix_scale=pix_scale
            )

        runit = pars['r_unit']
        vunit = pars['v_unit']

        if normalized is True:
            cstr = ' / c'
        else:
            cstr = ''

        if speed is True:
            mtype = 'Speed'
        else:
            mtype = 'Velocity'

        plt.pcolormesh(x, y, V)
        plt.colorbar(label=f'{vunit}{cstr}')

        runit = pars['r_unit']
        plt.xlabel(f'{runit}')
        plt.ylabel(f'{runit}')

        if center is True:
            plt.plot(0, 0, 'r', ms=10, markeredgewidth=2, marker='x')

        if title is not None:
            plt.title(title)
        else:
            rscale = pars['rscale']
            plt.title(f'{mtype} map in {plane} plane\n' +\
                      f'r_0 = {rscale} {runit}')

        if size is not None:
            plt.gcf().set_size_inches(size)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

    def plot_all_planes(
            self, plot_kwargs={}, show=True, close=True, outfile=None,
            size=(13, 8), speed=False, normalized=False, center=True
            ):
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
            rmax = 5. * pars['rscale']
            plot_kwargs['rmax'] = rmax

        # Create square grid centered at source for all planes
        Nr = 100
        dr = rmax / (Nr+1)
        r = np.arange(0, rmax+dr, dr)

        dx = (2*rmax) / (2*Nr+1)
        x = np.arange(-rmax, rmax+dx, dx)

        X, Y = np.meshgrid(x, x, indexing='ij')

        # Figure out subplot grid
        Nplanes = len(self._planes)

        if Nplanes == 4:
            ncols = 2
        else:
            ncols = 3

        k = 1
        for plane in self._planes:
            plt.subplot(2,ncols,k)
            # X, Y = transform.transform_coords(Xdisk, Ydisk, 'disk', plane, pars)
            # X,Y = Xdisk, Ydisk
            # Xplane, Yplane = transform.transform_coords(X,Y, plane)
            # X,Y = transform.transform_coords(Xdisk, Ydisk, 'disk', 'obs', pars)
            self.plot(plane, x=X, y=Y, show=False, close=False, speed=speed,
                      normalized=normalized, center=center, **plot_kwargs)
            k += 1

        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

    def plot_map_transforms(
            self, size=None, outfile=None, show=True, close=True, speed=False, 
            center=True, rmax=None, X=None, Y=None, pix_scale=None
            ):
        '''
        X, Y: np.ndarray (2d)
            Original grid positions to evaluate the velocity map on, in the disk plane. Defaults to (-rmax, rmax) in 100 steps
        rmax: float
            Maximum rendering distance from disk center. Defaults to 5 times the rscale of the vmap
        '''

        pars = self.model.pars

        runit = pars['r_unit']
        vunit = pars['v_unit']
        rscale = pars['rscale']

        if (runit == 'arcsec') and (pix_scale is None):
            raise ValueError('Must pass pix_scale if r_unit is arcsec!')

        if rmax is None:
            rmax = 5. * rscale

        Nr = 100

        if (X is None) and (Y is None):
            Nx = 2*Nr + 1
            dx = 2*rmax / Nx
            x = np.arange(-rmax, rmax+dx, dx)
            y = np.arange(-rmax, rmax+dx, dx)

            # uniform grid in disk plane
            X, Y = np.meshgrid(x, y, indexing='ij')
            xlim, ylim = [-rmax, rmax], [-rmax, rmax]

        else:
            xlim = [np.min(X), np.max(X)]
            ylim = [np.min(Y), np.max(Y)]

        if speed is True:
            mtype = 'Speed'
        else:
            mtype = 'Velocity'

        # don't bother plotting cen plane if there is no offset
        planes = self._planes.copy()
        if self.model.has_offset is False:
            planes.remove('cen')

        if len(planes) == 4:
            ncols = 2
            size = (9,8)
        else:
            ncols = 3
            size = (13,8)

        for k, plane in enumerate(planes):
            Xplane, Yplane = transform.transform_coords(X, Y, 'disk', plane, pars)
            Vplane = self(plane, Xplane, Yplane, speed=speed)

            plt.subplot(2,ncols,k+1)
            plt.pcolormesh(Xplane, Yplane, Vplane)
            plt.colorbar(label=f'{vunit}')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel(f'{runit}')
            plt.ylabel(f'{runit}')
            plt.title(f'{mtype} map for {plane} plane\n' +\
                      f'r_scale = {rscale} {runit}')

            if center is True:
                plt.plot(0, 0, 'r', ms=10, markeredgewidth=2, marker='x')

        plt.gcf().set_size_inches(size)
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return
    
    def plot_rotation_curve(
            self,
            X, # 2D grid of x-coordinates; pixels
            Y, # 2D grid of y-coordinates; pixels
            plane='obs',
            mask=None,
            pix_scale=None, # arcsec/pixel
            scale_radius=None, # arcsec
            threshold_dist=5,
            Nrbins=20,
            s=(12,5),
            print_model=True,
            show=True,
            out_file=None
            ):
        '''
        Plot the rotation curve of the velocity map for a given X, Y grid

        Parameters
        ----------
        TODO
        '''

        if mask is None:
            mask = np.ones_like(X, dtype=bool)

        model_pars = self.model.pars
        r_unit = model_pars['r_unit']
        rscale = model_pars['rscale']

        if (r_unit == 'arcsec') and (pix_scale is None):
            raise ValueError('Must pass pix_scale if r_unit is arcsec!')

        if isinstance(pix_scale, units.Quantity):
            pix_scale = pix_scale.value
        if r_unit == 'arcsec':
            rscale_pix = rscale / pix_scale
        else:
            rscale_pix = rscale

        vmap_plane = self(plane, X, Y, pix_scale=pix_scale)

        # determine the distance from the major axis for all pixels
        if self.model_name == 'offset':
            x0, y0 = model_pars['x0'], model_pars['y0']
        else:
            x0, y0 = 0., 0.

        pa = OrientedAngle(
            model_pars['theta_int'], unit='rad', orientation='cartesian'
        )
        vcirc = model_pars['vcirc']

        dist_major = dist_to_major_axis(X, Y, x0, y0, pa)

        # bin the velocity map by radius from the galaxy center
        dx = np.cos(pa)
        dy = np.sin(pa)
        R_signed = (X - x0) * dx + (Y - y0) * dy

        # now find the pixels within the threshold distance of the major axis
        # dist_major = np.abs(Y_rot)
        R_in_major = R_signed[(dist_major <= threshold_dist) & mask]

        if len(R_in_major) == 0:
            raise ValueError(
                f'No pixels within {threshold_dist} of the major axis; '
                )
        Rmin = R_in_major.min()
        Rmax = R_in_major.max()
        rbins = np.linspace(Rmin, Rmax, Nrbins+1)

        bin_centers = (rbins[:-1] + rbins[1:]) / 2
        rotation_curve = []
        rotation_curve_err = []

        # Calculate average velocity for each bin
        for n in range(Nrbins):
            radial_mask = (R_signed >= rbins[n]) & (R_signed < rbins[n+1])

            # now apply the other mask
            dist_major_mask = dist_major <= threshold_dist

            # combine the masks
            combined_mask = radial_mask & dist_major_mask

            if np.any(combined_mask):
                sini = model_pars['sini']
                avg_obs_vel = np.mean(vmap_plane[combined_mask] / sini)
                std_obs_vel = np.std(vmap_plane[combined_mask] / sini)
                rotation_curve.append(avg_obs_vel)
                rotation_curve_err.append(std_obs_vel)
            else:
                # handle empty bins
                rotation_curve.append(np.nan)
                rotation_curve_err.append(np.nan)

        plt.subplot(121)
        vmin, vmax = np.percentile(vmap_plane, 1), np.percentile(vmap_plane, 99)
        plt.imshow(
            vmap_plane, origin='lower', cmap='RdBu',
            norm=MidpointNormalize(vmin, vmax)
            )
        plot_line_on_image(
            vmap_plane, (x0, y0), pa, plt.gca(), c='k', label='model major axis'
        )
        # plot the threshold region
        plot_line_on_image(
            vmap_plane, (x0, y0+threshold_dist/2.), pa, plt.gca(), c='k', 
            ls=':', label='model major axis'
        )
        plot_line_on_image(
            vmap_plane, (x0, y0-threshold_dist/2.), pa, plt.gca(), c='k', 
            ls=':', label='model major axis'
        )
        plt.xlim(0, len(X))
        plt.ylim(0, len(Y))
        plt.title(f'Model Velocity Map ({plane} plane)')

        if print_model is True:
            i = 0
            for name, val in model_pars.items():
                if isinstance(val, float):
                    val = f'{val:.3f}'
                plt.text(
                    0.1,
                    0.9-i,
                    f'{name}: {val}', transform=plt.gca().transAxes
                    )
                i += 0.05

        plt.subplot(122)
        plt.errorbar(
            bin_centers, rotation_curve, rotation_curve_err, marker='o', label='model'
                    )
        plt.axhline(vcirc, c='k', ls='--', label='vcirc')
        plt.axhline(-vcirc, c='k', ls='--')
        plt.axvline(rscale_pix, c='g', ls=':', label='rscale')
        plt.axvline(-rscale_pix, c='g', ls=':')
        if scale_radius is not None:
            if pix_scale is None:
                v22 = self.compute_v22(scale_radius, 1)
                scale_radius_pix = scale_radius
            else:
                v22 = self.compute_v22(scale_radius, pix_scale)
                scale_radius_pix = scale_radius / pix_scale
            plt.axvline(
                scale_radius_pix, c='orange', ls='--', label='scale_radius'
                )
            plt.axhline(v22, c='purple', ls=':', label='v22')
        plt.xlabel('Radial Distance (pixels)')
        plt.ylabel('2D Rotational Velocity (km/s)')
        plt.legend()
        plt.title(f'Model Rotation Curve')
        plt.grid(True)

        plt.gcf().set_size_inches(s)

        if out_file is not None:
            save = True
        plot(show, save, out_file=out_file)

        return

    def compute_v22(self, scale_radius, pix_scale):
        '''
        Compute the V_22 parameter for the velocity model given the scale radius
        in arcsec and pixel scale in arcsec/pixel. V_22 is defined as the velocity
        at 2.2 * scale_radius.

        Parameters
        ----------
        scale_radius: float
            The scale radius of the model in arcsec
        pix_scale: float
            The pixel scale of the model grid in arcsec/pixel
        '''

        if isinstance(scale_radius, units.Quantity):
            scale_radius = scale_radius.value
        if isinstance(pix_scale, units.Quantity):
            pix_scale = pix_scale.value

        # find the position of the velocity map that we need to sample at
        theta_int = self.model.pars['theta_int']
        scale_radius_pix = scale_radius / pix_scale
        x = scale_radius_pix * np.cos(theta_int)
        y = scale_radius_pix * np.sin(theta_int)

        # for annoying reasons, this must be done
        x = np.array([x])
        y = np.array([y])

        # evaluate the velocity map at this position
        r_unit = self.model.pars['r_unit']
        if r_unit == 'arcsec':
            v22 = self('obs', x, y, pix_scale=pix_scale)
        else:
            v22 = self('obs', x, y)

        return v22

def get_model_types():
    return MODEL_TYPES

# NOTE: This is where you must register a new model
MODEL_TYPES = {
    'default': VelocityModel,
    'centered': VelocityModel,
    'offset': OffsetVelocityModel
    }

def build_model(name, pars, logger=None):
    name = name.lower()

    if name in MODEL_TYPES.keys():
        # User-defined input construction
        model = MODEL_TYPES[name](pars)
    else:
        raise ValueError(f'{name} is not a registered velocity model!')

    return model

#-------------------------------------------------------------------------------
# some utility methods

def dist_to_major_axis(X, Y, x0, y0, position_angle):
    '''
    Calculate the distance from each pixel to the major axis

    Parameters
    ----------
    X : np.ndarray
        2D array of x-coordinates
    Y : np.ndarray
        2D array of y-coordinates
    x/y0 : float
        x/y-coordinate of the galaxy center
    position_angle : OrientedAngle
        Position angle of the galaxy major axis
    '''

    # get distance from each pixel to the galaxy center
    Xcen, Ycen = X - x0, Y - y0
    R = np.sqrt(Xcen**2 + Ycen**2)

    # angle from each pixel to the galaxy center
    Theta = np.arctan2(Ycen, Xcen) # returns (-pi, pi)

    # match the [0, 360] wrapping of the position angle
    Theta[Theta < 0] += 2 * np.pi 

    # difference in angle between the pixel and the major axis
    dTheta = position_angle.cartesian.rad - Theta

    # calculate distance from each pixel to the major axis
    dist = abs(R * np.sin(dTheta))

    # import matplotlib.colors as mcolors
    # norm = mcolors.TwoSlopeNorm(
    #     vmin=-np.max(np.abs(dist)), vcenter=0, vmax=np.max(np.abs(dist))
    #    )
    # plt.imshow(dist.T, origin='lower', cmap='RdBu', norm=norm)
    # plt.text(0.1, 0.1, f'PA: {np.rad2deg(position_angle):.2f} deg')
    # plt.colorbar()
    # plt.show()

    return dist
