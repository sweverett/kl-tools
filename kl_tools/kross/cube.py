'''
This file contains methods & classes related to the ingestion of KROSS data 
into core kl_tools data structures such as a DataCube subclass.
'''

import numpy as np
import astropy.units as u
from astropy.io import fits
from reproject import reproject_interp
import astropy.wcs as wcs
import galsim as gs
import warnings

from kl_tools.cube import DataCube, CubePars
from kl_tools.emission import EmissionLine, LINE_LAMBDAS
from kl_tools.kross.data import get_kross_obj_data

class KROSSDataCube(DataCube):
    '''
    An interface to construct a DataCube object from KROSS data.
    '''

    def __init__(self, kid):

        #-----------------------------------------------------------------------
        # get the object data

        obj_data = get_kross_obj_data(kid)

        #-----------------------------------------------------------------------
        # setup the datacube parameters and instance

        cube_pars = self._setup_cube_pars(obj_data)

        super(KROSSDataCube, self).__init__(obj_data['cube'], pars=cube_pars)

        #-----------------------------------------------------------------------
        # extra KROSS-specific data

        self.kross_files = {
            'cube': obj_data['cube_file'],
            'velocity': obj_data['velocity_file'],
            'sigma': obj_data['sigma_file'],
            'halpha': obj_data['halpha_file'],
            'hst': obj_data['hst_file']
        }

        self.kid = kid
        self.name = obj_data['catalog']['NAME'][0].strip()
        self.obj_data = obj_data['catalog']

        # set the masks and weights
        # NOTE: needs the self.kross_files info to be set first
        self.set_weights()
        self.set_masks()

        return

    def _setup_cube_pars(self, obj_data, sigmas=7):
        '''
        Setup the parameters for the KROSS datacube.

        Parameters
        ----------
        obj_data : dict
            A dictionary containing the object data from get_kross_obj_data()
        sigmas : int, optional. Default=5
            The number of instrumental emission line broadening sigmas to use 
            for the bandpass limits
        '''

        cube_hdr = obj_data['cube_hdr']

        CRVAL3 = float(cube_hdr['CRVAL3'])
        CDELT3 = float(cube_hdr['CDELT3'])
        NAXIS3 = int(cube_hdr['NAXIS3'])

        wavelengths = 1e3 * (CRVAL3 + CDELT3 * np.arange(NAXIS3)) # nm
        wav_blue = wavelengths - CDELT3/2.
        wav_red = wavelengths + CDELT3/2.

        bandpasses = [gs.Bandpass(
            1.0,
            'nm', # must be nm or A
            blue_limit=wav_blue[i],
            red_limit=wav_red[i],
            zeropoint=22.5
            ) for i in range(len(wavelengths))
            ]

        # some are set later
        specs = {
            # some approximations from the KMOS spec sheet
            # https://www.eso.org/sci/facilities/paranal/instruments/kmos/inst.html
            # 'throughput': ...,
            'resolution': 3582., # band center for YJ bandpass
        }

        line_names = ['Ha'] # just Halpha for now

        z = obj_data['catalog']['Z'][0]
        R = specs['resolution']

        lines = []
        for i, line_name in enumerate(line_names):
            # TODO: We should investigate whether to use LAMBDA_SN directly,
            # or even marginalize over z
            line_lambda = LINE_LAMBDAS[line_name]

            line_pars = {
                'name': line_name,
                'value': line_lambda.value,
                'R': R,
                'z': z,
                'unit': line_lambda.unit
            }

            sed_pars = {
                # +/- 3 sigma using central R value
                'lblue': line_lambda.value * (1 + z) * (1 - sigmas / (2.355*R)),
                'lred': line_lambda.value * (1 + z) * (1 + sigmas / (2.355*R)),
                'resolution': 0.1, # angstroms
                'unit': line_lambda.unit
            }
            lines.append(EmissionLine(line_pars, sed_pars))

        pars_dict = {
            # see https://astro.dur.ac.uk/KROSS/data.html
            'pix_scale': 0.1 * u.arcsec, # square
            'bandpasses': bandpasses,
            'emission_lines': lines,
        }

        pars = CubePars(pars_dict)

        return pars

    def set_masks(self, masks=None, method='vmap'):
        '''
        Set masks from a KROSS vmap datacube file. If no masks are explicitly
        provided, the mask will be inherited (and reprojected) from the velocity
        mask.

        If instead you'd like a simpler mask that only masks pixels that are 0
        or NaN, set method to `simple`

        NOTE: We use the convention that a pixel is masked if the mask is True
        for the pixel. That means selections should be done with data[~mask]

        Parameters
        ----------
        masks : np.ndarray, optional. Default=None
            A 2D or 3D array of masks to apply to the datacube. If None, the 
            masks will only be sensitive to pixels that are 0 or NaN
        method: str, optional. Default='vmap'
            The method to use to set the masks. Options are 'vmap' and 'simple'
        '''

        if masks is None:
            if method == 'vmap':
                # reproject the 2D velocity map mask and use for all slices

                # get the velocity map and needed WCS's
                cube_file = self.kross_files['cube']
                vmap_file = self.kross_files['velocity']
                # known issues in KROSS headers
                warnings.filterwarnings(
                    'ignore', category=wcs.FITSFixedWarning
                    )
                with fits.open(cube_file) as hdul:
                    cube_wcs = wcs.WCS(hdul[0].header)
                with fits.open(vmap_file) as hdul:
                    vmap_wcs = wcs.WCS(hdul[0].header)
                cube_slice_wcs = cube_wcs.dropaxis(2) # need 2D WCS

                with fits.open(vmap_file) as hdul:
                    vmap = hdul[0].data
                vmap_mask = np.zeros_like(vmap)
                vmap_mask[vmap == 0] = 1 # mask is "on" where velocity is zero

                # now reproject the velocity map to the cube WCS
                reprojected_mask, _ = reproject_interp(
                    (vmap_mask, vmap_wcs),
                    cube_slice_wcs,
                    shape_out=cube_slice_wcs.array_shape,
                    order='nearest-neighbor' # best for categorical data
                )
                masks = (reprojected_mask > 0.5).astype(bool)

            elif method == 'simple':
                # mask pixels that are 0 or NaN
                masks = np.zeros_like(self.data)
                masks[self.data == 0] = 0
                masks[np.isnan(self.data)] = 0
            else:
                raise ValueError(f'Unknown mask method: {method}')

        # will set all masks to the same mask
        super(KROSSDataCube, self).set_masks(masks)

        return

    def set_weights(self, weights=None):
        '''
        Set weights from the KROSS data files.

        NOTE: We have yet to find how the per-pixel weights are stored,
        so for now we will set all weights to 1.
        '''

        if weights is None:
            weights = np.ones_like(self.data)
        
        super(KROSSDataCube, self).set_weights(weights)

        return

    def set_continuum(self, lmin_line, lmax_line, lmin_cont, lmax_cont,
                      method='sum'):
        '''
        Build a 2d template for the continuum from the spectrum near the line.

        NOTE: This has been copied from muse.py, so let's re-examine if we can 
        refactor this into cube.py
        '''

        # Make a new cube object for the blue continuum and the red continuum.
        lblue = lmin_cont
        lred = lmin_line
        args, kwargs = self.truncate(lblue, lred, trunc_type='return-args')
        blue_cont_cube = DataCube(*args,**kwargs)
        blue_cont_template = np.sum(
            blue_cont_cube.data * blue_cont_cube.weights, axis=0
            ) / np.sum(blue_cont_cube.weights, axis=0)

        lblue = lmax_line
        lred = lmax_cont
        args, kwargs = self.truncate(lblue, lred, trunc_type='return-args')
        red_cont_cube = DataCube(*args,**kwargs)
        red_cont_template = np.sum(
            red_cont_cube.data * red_cont_cube.weights, axis=0
            ) / np.sum(red_cont_cube.weights, axis=0)

        self._continuum_template = (blue_cont_template + red_cont_template) / 2.

        return

    def set_line(self, line_choice='Ha', truncate=True):
        '''
        Set the emission line actually used in an analysis of a datacube,
        and possibly truncate it to slices near the line

        Parameters
        ----------
        line_choice : str
            The name of the emission line to use.
        '''

        emission_lines = [
            line.line_pars['name'] for line in self.pars['emission_lines']
        ]
        if line_choice == 'strongest':
            print(
                'Warning: using the strongest line is not officially supported '
                'in KROSS data. Using Halpha instead.'
            )
            line_choice = 'halpha'
        elif line_choice not in emission_lines:
            raise ValueError(
                f'The line choice {line_choice} is not in the list of emission '
                'lines'
            )

        line_index = [
            i for i, line in enumerate(emission_lines) if line == line_choice
        ][0]

        # update all line-related meta data
        self.pars['emission_lines'] = [self.pars['emission_lines'][line_index]]


        # create new truncated datacube around line, if desired
        if truncate is True:

            # Estimate a continuum template
            line = self.pars['emission_lines'][0]
            lu = line.line_pars['unit']
            lblue = line.sed_pars['lblue'] * lu
            lred = line.sed_pars['lred'] * lu
            boxwidth = lred - lblue

            self.set_continuum(lblue, lred, lblue-boxwidth, lred+boxwidth)

            # have to store these temporarily, and then set it after truncation
            kid = self.kid
            name = self.name
            kross_files = self.kross_files
            obj_data = self.obj_data
            continuum_template = self._continuum_template

            args, kwargs = self.truncate(lblue, lred, trunc_type='return-args')
            super(KROSSDataCube, self).__init__(*args, **kwargs)

            self.kid = kid
            self.name = name
            self.kross_files = kross_files
            self.obj_data = obj_data
            self._continuum_template = continuum_template

        return

    @property
    def z(self) -> float:
        return self.obj_data['Z'][0]