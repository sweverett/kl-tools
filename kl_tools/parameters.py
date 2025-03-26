from copy import deepcopy
import kl_tools.utils as utils
import os
import galsim as gs
import numpy as np
import astropy.units as u

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

    def __init__(self, sampled_pars, meta_pars, derived_pars={}):
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
            'meta_pars': (meta_pars, dict),
            'derived_pars': (derived_pars, dict),
            }
        utils.check_types(args)

        for name in sampled_pars:
            utils.check_type(name, 'sampled_par_val', str)

        pars_order = dict(zip(sampled_pars, range(len(sampled_pars))))
        self.sampled = SampledPars(pars_order)
        self.meta = MCMCPars(meta_pars)
        self.derived = DerivedPars(derived_pars)

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

    def get(self, key):
        return self.pars[key]

    def get(self, key, default):
        return self.pars.get(key, default)

    def pop(self, key):
        return self.pars.pop(key)

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

class DerivedPars(MetaPars):

    '''
    Class that defines structure for the derived parameters
    NOTE: 
        The input pars is a dict of derived param name: function string of 
        sampled parameters, e.g.
        pars = {"V22": "vcirc/(np.pi/2)*np.arctan(1.3*hlr/rscale)"}
    '''

    _req_fields = []
    _opt_fields = []
    def eval(self, key, **kwargs):
        ''' Evaluate the derived parameter 
        Input:
            - key: string
                Which derived parameter to evaluate
            - kwargs: **dict
                Sampled parameters used to evaluate the derived parameters
        '''
        print(kwargs)
        func_string = self.pars[key]
        res = eval(func_string)
        return res

########## parameters for observation and cube modeling ########

class CubePars(MetaPars):
    '''
    Class that defines structure for DataCube meta parameters,
    e.g. image & emission line meta data
    '''

    _req_fields = ['pix_scale', 'bandpasses']
    _opt_fields = ['psf', 'emission_lines', 'files', 'shape', 'meta', 'truth']
    _gen_fields = ['wavelengths'] # not an allowed input field, but generated

    def __init__(self, pars):
        '''
        pars: dict
            Dictionary of meta pars for the DataCube
        '''

        super(CubePars, self).__init__(pars)

        self._bandpasses = None
        bp = self.pars['bandpasses']
        try:
            utils.check_type(bp, 'bandpasses', list)
        except TypeError:
            try:
                utils.check_type(bp, 'bandpasses', dict)
            except:
                raise TypeError('CubePars bandpass field must be a' +\
                                'list of galsim bandpasses or a dict!')

        self.build_bandpasses()

        self._lambdas = None
        self._lambda_unit = None
        self.build_wavelength_list()

        return

    def build_bandpasses(self, remake=False):
        '''
        Build a bandpass list from pars if not provided directly
        '''

        # sometimes it is already set in the parameter dict
        if (remake is False) & (self._bandpasses is not None):
            return

        bp = self.pars['bandpasses']

        if isinstance(bp, list):
            # we'll be lazy and just check the first entry
            if not isinstance(bp[0], gs.Bandpass):
                raise TypeError('bandpass list must be filled with ' +\
                                'galsim.Bandpass objects!')
            bandpasses = bp
        else:
            # make this a separate dict
            bp_dict = deepcopy(bp)

            # already checked it is a list or dict
            bandpass_req = ['lambda_blue', 'lambda_red', 'dlambda']
            bandpass_opt = ['throughput', 'zp', 'unit', 'file']
            utils.check_fields(bp_dict, bandpass_req, bandpass_opt)

            args = [
                bp_dict.pop('lambda_blue'),
                bp_dict.pop('lambda_red'),
                bp_dict.pop('dlambda')
                ]

            kwargs = bp_dict

            bandpasses = setup_simple_bandpasses(*args, **kwargs)

        self._bandpasses = bandpasses
        self['bandpasses'] = bandpasses

        return

    def build_wavelength_list(self):
        '''
        Build a list of slice wavelength bounds (lambda_blue, lambda_red)
        given a set bandpass list
        '''

        if self._bandpasses is None:
            self.build_bandpasses()

        # Not necessarily needed, but could help ease of access
        self._lambda_unit = u.Unit(self._bandpasses[0].wave_type)
        self._lambdas = [] # Tuples of bandpass bounds in unit of bandpass
        for bp in self._bandpasses:
            # galsim bandpass limits are always stored in nm
            li = (bp.blue_limit * u.nm).to(self._lambda_unit).value
            le = (bp.red_limit * u.nm).to(self._lambda_unit).value
            self._lambdas.append((li, le))

            # Make sure units are consistent
            # (could generalize, but not necessary)
            assert bp.wave_type == self._lambda_unit
        self._lambdas = np.array(self._lambdas)

        self['wavelengths'] = self._lambdas

        return

    def reset(self):
        '''
        Reset any attributes or fields that were set during instantiation,
        not as a direct field input. Useful for datacube truncations
        '''

        self._bandpasses = None
        self._lambdas = None
        self._lambda_unit = None

        # remove any generated fields
        for field in self._gen_fields:
            del self[field]

        self.__init__(self.pars)

        return

    def set_shape(self, shape):
        '''
        Store the datacube shape in CubePars

        shape: (Nspec, Nx, Ny) tuple
        '''

        if len(shape) != 3:
            raise ValueError('shape must be a tuple of len 3!')

        # this will build a lambda list from the bandpasses, if not yet created
        lambdas = self.lambdas

        if shape[0] != len(lambdas):
            raise ValueError('The first dimension of shape must be the ' +\
                             'length as the bandpass & wavelength list!')

        self['shape'] = shape

        return

    @property
    def bandpasses(self):
        if self._bandpasses is None:
            self.build_bandpasses()

        return self._bandpasses

    @property
    def lambdas(self):
        if self._lambdas is None:
            self.build_wavelength_list()

        return self._lambdas

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return CubePars(deepcopy(self.pars))


class FiberPars(MetaPars):
    _req_fields = ['model_dimension']
    # model_dimension:
    #   Nx, Ny, scale, lambda_range, lambda_res, lambda_unit

    _opt_fields = ['obs_conf',]
    # obs_conf:
    #   If provided, overwrite the obs_conf in pars, See grism.GrismDataVector.data_header

    _gen_fields = ['pix_scale', 'shape']
    # bandpasses:
    #   lambda_blue, lambda_red, dlambda, throughput, zp, unit, file

    def __init__(self, meta_pars, obs_conf=None):
        ''' Fiber-related `Pars` obj
        Each `FiberPars` store the necessary parameters for one observation.

        Inputs:
        =======
        meta_pars: dict obj
            Dictionary of meta pars for `FiberCube`. Note that this dict has
            different convention than the parent class `cube.CubePars`. The
            initialization step will translate the dictionary.
        '''
        # assure the input dictionary obj has required fields & obs config
        self._check_pars(meta_pars)
        _pars = deepcopy(meta_pars) # avoid shallow-copy of a dict object
        if obs_conf is None and _pars.get('obs_conf', None) is None:
            raise ValueError('Observation configuration is not set!')
        if obs_conf is not None:
            utils.check_type(obs_conf, 'obs_conf', dict)
            _pars['obs_conf'] = deepcopy(obs_conf)
        super(FiberPars, self).__init__(_pars)
        self.is_dispersed = (self.pars['obs_conf']['OBSTYPE'] == 1)

        # set up bandpass, model cube shape, and model cube pixel scale

        ### 1. Fiber Data

        if self.is_dispersed:
            _bid = self.conf['SEDBLKID']
            _lrange = self.pars['model_dimension']['lambda_range'][_bid]
            _dlam = _pars['model_dimension']['lambda_res']
            if isinstance(_dlam, list):
                _dlam = _dlam[_bid]
            self.throughput = gs.Bandpass(self.pars['obs_conf']['BANDPASS'],
                'nm', blue_limit=_lrange[0], red_limit=_lrange[1])
            self.wave = np.arange(_lrange[0], _lrange[1], _dlam)
            self._bp_array = self.throughput(self.wave)
            self.lambdas = np.array([self.wave, self.wave + _dlam]).T
            _Nlam = len(self.wave)

        ### 2. Photometry data

        else:
            _from_file_ = os.path.isfile(self.pars['obs_conf']['BANDPASS'])
            if _from_file_:
                self.throughput = gs.Bandpass(self.pars['obs_conf']['BANDPASS'], 'nm')
            else:
                _lrange = [np.min(self.pars['model_dimension']['lambda_range']),
                        np.max(self.pars['model_dimension']['lambda_range'])]
                self.throughput = gs.Bandpass(self.pars['obs_conf']['BANDPASS'],
                'nm', blue_limit=_lrange[0], red_limit=_lrange[1])
            self.wave, _Nlam, self._bp_array = None, 1, None

        self.pars['shape'] = np.array([_Nlam,
                         self.pars['model_dimension']['Ny'],
                         self.pars['model_dimension']['Nx']], dtype=int)
        self.pars['pix_scale'] = self.pars['model_dimension']['scale']
        self.X, self.Y = utils.build_map_grid(self.pars['shape'][2],
            self.pars['shape'][1], indexing='xy', scale=self.pars['pix_scale'])
        self.obs_index = self.pars['obs_conf']['OBSINDEX']

        return

    # quick approach to the `obs_conf` observation configuration (1 obs)
    @property
    def conf(self):
        return self.pars['obs_conf']

    # get the Nx2 band-pass array, which has the same dimension to `lambdas`
    @property
    def bp_array(self):
        # self._bp_array = np.zeros(self.lambdas.shape)
        # for i,_bp in enumerate(self.bandpasses):
        #     self._bp_array[i][0] = _bp(self.lambdas[i][0])
        #     self._bp_array[i][1] = _bp(self.lambdas[i][1])
        return self._bp_array

    @property
    def lambda_eff(self):
        return self.throughput.effective_wavelength

def merge_bandpasses(bandpasses):
    blim, rlim = bandpasses[0].blue_limit, bandpasses[-1].red_limit
    wtype, zp = bandpasses[0].wave_type, bandpasses[0].zeropoint
    waves = [(bp.blue_limit+bp.red_limit)/2. for bp in bandpasses]
    trans = [bp(w) for bp,w in zip(bandpasses, waves)]
    table = gs.LookupTable(waves, trans)
    bandpass = gs.Bandpass(table, wave_type=wtype, zeropoint=zp)
    return bandpass

def setup_simple_bandpasses(lambda_blue, lambda_red, dlambda,
                            throughput=1., zp=30., unit='nm', file=None):
    '''
    Setup list of bandpasses needed to instantiate a DataCube
    given the simple case of constant spectral resolution, throughput,
    and image zeropoints for all slices

    Useful for quick setup of tests and simulated datavectors

    lambda_blue: float
        Blue-end of datacube wavelength range
    lambda_red: float
        Rd-end of datacube wavelength range
    dlambda: float
        Constant wavelength range per slice
    throughput: float
        Throughput of filter of data slices
    unit: str
        The wavelength unit
    zeropoint: float
        Image zeropoint for all data slices
    '''

    li, lf = lambda_blue, lambda_red
    lambdas = [(l, l+dlambda) for l in np.arange(li, lf, dlambda)]

    bandpasses = []
    if file is None:
        for l1, l2 in lambdas:
            bandpasses.append(gs.Bandpass(f'{throughput}', unit,
                blue_limit=l1, red_limit=l2, zeropoint=zp))
    else:
        # build bandpass from file
        assert os.path.exists(file), f'Bandpass file {file} does not exist!'
        _bandpass = gs.Bandpass(file, wave_type = unit)
        for l1, l2 in lambdas:
            bandpasses.append(_bandpass.truncate(blue_limit=l1, red_limit=l2))

    return bandpasses
