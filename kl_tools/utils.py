import numpy as np
import os, sys
from pathlib import Path
import yaml
import matplotlib.colors as colors
import pdb
import copy
import kl_tools.priors as priors
try:
    import schwimmbad
    import mpi4py
    from mpi4py import MPI
except:
    print("Can not import MPI, use single process")
try:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    rank = 0
    size = 1

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

        return

class MidpointNormalize(colors.Normalize):
    '''
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    '''

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

        return

    def __call__(self, value, clip=None):
        # Ignoring masked values and edge cases
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        return yaml.safe_load(stream)

def build_map_grid(Nx, Ny, indexing='ij', scale=1.0):
    '''
    We define the grid positions as the center of pixels

    For a given dimension: if # of pixels is even, then
    image center is on pixel corners. Else, a pixel center
    '''

    # max distance in given direction
    # even pixel counts requires offset by 0.5 pixels
    Rx = (Nx // 2) - 0.5 * ((Nx-1) % 2)
    Ry = (Ny // 2) - 0.5 * ((Ny-1) % 2)

    x = np.arange(-Rx, Rx+1, 1) * scale
    y = np.arange(-Ry, Ry+1, 1) * scale

    assert len(x) == Nx
    assert len(y) == Ny

    X, Y = np.meshgrid(x, y, indexing=indexing)

    return X, Y

def check_file(filename):
    '''
    Check if file exists; err if not
    '''

    if not os.path.exists(filename):
        raise OSError(f'{filename} does not exist!')

    return

def check_type(var, name, desired_type):
    '''
    Checks that the passed variable (with given name)
    is of the desired type
    '''

    if not isinstance(var, desired_type):
        raise TypeError(f'{name} must be a {desired_type}!')

    return

def check_types(var_dict):
    '''
    Check that the passed variables match the desired type.
    Convenience wrapper around check_type() for multiple variables

    var_dict: dict
        A dictionary in the format of name: (var, desired_type)
    '''

    for name, tup in var_dict.items():
        var, desired_type = tup
        check_type(var, name, desired_type)

    return

def check_req_fields(config, req, name=None):
    for field in req:
        if not field in config:
            raise ValueError(f'{name}config must have field {field}')

    return

def check_fields(config, req, opt, name=None):
    '''
    req: list of required field names
    opt: list of optional field names
    name: name of config type, for extra print info
    '''
    assert isinstance(config, dict)

    if name is None:
        name = ''
    else:
        name = name + ' '

    if req is None:
        req = []
    if opt is None:
        opt = []

    # ensure all req fields are present
    check_req_fields(config, req, name=name)

    # now check for fields not in either
    for field in config:
        if (not field in req) and (not field in opt):
            raise ValueError(f'{field} not a valid field for {name} config!')

    return

def make_dir(d):
    '''
    Makes dir if it does not already exist
    '''

    # if not os.path.exists(d):
    #     os.mkdir(d)
    Path(d).mkdir(parents=True, exist_ok=True)

    return

def get_base_dir():
    '''
    base dir is parent repo dir
    '''
    module_dir = get_module_dir()
    return os.path.dirname(module_dir)

def get_module_dir():
    return os.path.dirname(__file__)

def get_test_dir():
    base_dir = get_base_dir()
    test_dir = os.path.join(base_dir, 'tests')
    make_dir(test_dir) # will only create if it does not exist
    return test_dir

def set_cache_dir():
    basedir = get_base_dir()
    cachedir = os.path.join(basedir, '.cache/')
    make_dir(cachedir) # will only create if it does not exist
    return cachedir

def parse_param(config, params):
    Nsampled = 0
    for key in config.keys():
        if type(config[key])==dict:
            Nsampled += parse_param(config[key], params)
        else:
            if (config[key] == 'param')|(config[key] == 'Param')|(config[key] == 'PARAM'):
                if "value" in params[key].keys():
                    config[key] = params[key]['value']
                elif "prior" in params[key].keys():
                    config[key] = 'sampled'
                    Nsampled += 1
                else:
                    if rank==0:
                        print("WARNING: can not parse parameter %s in meta!"%key)
    return Nsampled

def parse_prior(config):
    prior_type = config.get("dist", "flat")
    if prior_type=="flat":
        return priors.UniformPrior(config["min"], config["max"],
                                   inclusive=config.get("inclusive", False))
    elif prior_type=="norm":
        return priors.GaussPrior(config["loc"], config["scale"],
                                 clip_sigmas=config.get("clip_sigmas", None),
                                 zero_boundary=config.get("zero_boundary", None))
    elif prior_type=="lognorm":
        return priors.LognormalPrior(config["mu"], config["dex"],
                                   clip_sigmas=config.get("clip_sigmas", None))
    else:
        raise ValueError(f'Can not support prior of type {prior_type}!')

def parse_propose(config):
    # parse propose
    if config["dist"] == "norm":
        mean, std, proposal = config["loc"], config["scale"], config["proposal"]
    else:
        raise ValueError(f'Proposal distribution {config["dist"]} is not supported now!')
    return mean, std, proposal

def parse_yaml(filename):
    ''' Parse the YAML configuration file for a KL fitting
    Input:
        filename: YAML file name
    Output:
        sampled_pars: list of string
            which parameters to sample
        fidvals: dict
            fiducial values of those sampled parameters
        latex_labels: dict
            the LaTeX labels of those parameters
        derived: dict
            derived parameters
        meta: dict
            meta parameter dict
        mcmc: dict
            MCMC settings
        sample_ball_mean: dict
            mean values of the sampled parameters, used in MCMC initialization
        sample_ball_std: dict
            std values of the sampled parameters, used in MCMC initialization
        sample_ball_proposal: dict
            proposal of MCMC
    '''
    config = read_yaml(filename)
    sampled_pars = []
    sample_ball_mean, sample_ball_std, sample_ball_proposal = {}, {}, {}
    fidvals = {}
    latex_labels = {}
    derived = {}
    # make a list of fiducial values of parameters being sampled
    for key in config['params']:
        pinfo = config['params'][key]
        # sampled parameters with prior and is not derived params
        if ("prior" in pinfo.keys()) and ("derived" not in pinfo.keys()):
            sampled_pars.append(key)
            fidvals[key] = config['params'][key]["ref"]["loc"]
        # setup derived parameters
        if "derived" in pinfo.keys():
            derived[key] = config['params'][key]["derived"]
        latex_labels[key] = config['params'][key].get("latex", key)
    # start from the `meta` field and fill parameters that are set to `param`
    Nsampled = parse_param(config['meta'], config['params'])
    if rank==0:
        print(f'{Nsampled} parameters are being sampled: {sampled_pars}')
    ### set the priors in meta
    ### Incl. sampled AND derived parameters
    config['meta']['priors'] = {}
    for key in sampled_pars:
        config['meta']['priors'][key] = parse_prior(config['params'][key]['prior'])
    for key in derived.keys():
        pinfo = config["params"][key]
        if "prior" in pinfo.keys():
            config['meta']['priors'][key] = parse_prior(pinfo['prior'])
    # set the MCMC initialized sample ball
    for key in sampled_pars:
        propose_mean, propose_std, proposal = parse_propose(config["params"][key]["ref"])
        sample_ball_mean[key] = propose_mean
        sample_ball_std[key] = propose_std
        sample_ball_proposal[key] = proposal
    # tune the observation configuration
    Nobs = len(config['meta']['obs_conf'])
    for i in range(Nobs):
        config['meta']['obs_conf'][i]['OBSINDEX'] = i
        if config['meta']['obs_conf'][i]['OBSTYPE'] == 'fiber':
            config['meta']['obs_conf'][i]['OBSTYPE'] = 1
        elif config['meta']['obs_conf'][i]['OBSTYPE'] == 'grism':
            config['meta']['obs_conf'][i]['OBSTYPE'] = 2
        elif config['meta']['obs_conf'][i]['OBSTYPE'] == 'image':
            config['meta']['obs_conf'][i]['OBSTYPE'] = 0
        else:
            raise ValueError("Unrecognized observation type %s!"%(config['meta']['obs_conf'][i]['OBSTYPE']))
    meta = copy.deepcopy(config['meta'])
    mcmc = copy.deepcopy(config['mcmc'])
    return sampled_pars, fidvals, latex_labels, derived, meta, mcmc, \
        sample_ball_mean, sample_ball_std, sample_ball_proposal


BASE_DIR = get_base_dir()
MODULE_DIR = get_module_dir()
TEST_DIR = get_test_dir()
CACHE_DIR = set_cache_dir()
