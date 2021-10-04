import numpy as np
import pudb

'''
This file contains transformation functions. These
are all static functions so that numba can be used
efficiently.

Definition of each plane:

    disk: Face-on view of the galactic disk, no inclination angle.
          This will be cylindrically symmetric for most models

    gal:  Galaxy major/minor axis frame with inclination angle same as
          source plane. Will now be ~ellipsoidal for sini!=0

    source: View from the lensing source plane, rotated version of gal
            plane with theta = theta_intrinsic

    obs:  Observed image plane. Sheared version of source plane
'''

class TransformableImage(object):
    '''
    This base class defines the transformation properties
    and structure for an image (e.g. velocity map, source
    intensity)that can be rendered in the various planes
    relevant to kinematic lensing: disk, gal, source, and obs

    The unifying feature of these images will be that their
    rendering in a given plane will be handled by the __call__
    attribute, along with the plane name, image coords to eval
    at, and any necessary transformation parameters
    '''

    def __init__(self, transform_pars):
        '''
        transform_pars: dict
            A dictionary that defines the parameters needed
            to evaluate the plane transformations
        '''

        self.transform_pars = transform_pars
        self._planes = ['disk', 'gal', 'source', 'obs']

        self._setup_transformations()

        return

    def _setup_transformations(self):
        '''
        TODO: Do we need this?
        '''

        pars = self.transform_pars

        self.obs2source = _transform_obs2source(pars)
        self.source2gal = _transform_source2gal(pars)
        self.gal2disk = _transform_gal2disk(pars)

        return

    def __call__(self, plane, x, y, use_numba=False):
        '''
        plane: str
            The plane to evaluate the image in
        x: np.ndarray
            The x position(s) to evaluate the image at
        y: np.ndarray
            The y position(s) to evaluate the image at
        use_numba: bool
            Whether to use numba-friendly versions of the
            transformation functions
            NOTE: Not yet fully implemented
        '''

        if plane not in self._planes:
            raise ValueError(f'{plane} not a valid image plane!')

        if not isinstance(x, np.ndarray):
            assert not isinstance(y, np.ndarray)
            x = np.array(x)
            y = np.array(y)

        if x.shape != y.shape:
            raise Exception('x and y arrays must be the same shape!')

        return

def transform_coords(x, y, plane1, plane2, pars):
    '''
    Transform coords (x,y) defined in plane1 into plane2

    pars: dict holding model information
    '''

    # each plane assigned an index in order from
    # simplest disk plane to most complex obs plane
    planes = ['disk', 'gal', 'source', 'obs']
    plane_map = dict(zip(planes, range(len(planes))))

    for plane in [plane1, plane2]:
        if plane not in planes:
            raise ValueError(f'{plane} not a valid plane!')

    # get start & end indices
    start = plane_map[plane1]
    end   = plane_map[plane2]

    if start == end:
        return x, y

    # transforms in direction from disk to obs
    if start < end:
        transforms = [_disk2gal, _gal2source, _source2obs]
        step = 1

    # transforms in direction from obs to disk
    else:
        transforms = [_gal2disk, _source2gal, _obs2source]
        step = -1

        # Account for different starting point for inv transforms
        start -= 1
        end -= 1

    # there is no transform starting with the end indx
    for i in range(start, end, step):
        transform = transforms[i]
        x, y = transform(pars, x, y)

    return x, y

def _multiply(transform, x, y):
    '''
    transform: a (2x2) coordinate transformation matrix
    x: np array of x position
    y: np array of y position

    returns: (x', y')
    '''

    # Can't just do a matrix multiplication because we can't assume
    # structure of pos; e.g. it might be a meshgrid array

    assert x.shape == y.shape

    # x,y are vectors
    if len(x.shape) == 1:
        pos = np.array([x, y])

        out = np.matmul(transform, pos)
        xp, yp = out[0], out[1]

    # x,y are matrices
    elif len(x.shape) == 2:
        # gotta do it the slow way
        # out = np.zeros((2, x.shape[0], x.shape[1]))
        xp = np.zeros(x.shape)
        yp = np.zeros(y.shape)

        N1, N2 = x.shape
        for i in range(N1):
            pos = np.array([x[i,:], y[i,:]])
            out = np.matmul(transform, pos)
            xp[i,:] = out[0]
            yp[i,:] = out[1]

    else:
        raise ValueError('Plane transformations are not yet implemented ' +\
                         f'for input positions of shape {x.shape}!')

    return xp, yp

# The following transformation definitions require basic knowledge
# of the source shear, intrinsic orientation, and profile inclination
# angle
#
# The inverse transforms are computed on the fly from these defs
# TODO: Could try to work out these inv transforms analytically
#       for faster eval

def _transform_obs2source(pars):
    '''
    Lensing transformation from obs to source plane

    pars is a dict
    (x,y) is position on obs plane
    '''

    g1, g2 = pars['g1'], pars['g2']

    # Lensing transformation
    transform =  np.array([
        [1.-g1, -g2],
        [-g2, 1.+g1]
    ])

    return transform

def _transform_source2obs(pars):
    '''
    Inverse lensing transformation from source to obs plane

    pars is a dict
    (x,y) is position on obs plane
    '''

    g1, g2 = pars['g1'], pars['g2']

    # Lensing transformation
    norm = 1. / (g1**2 + g2**2 - 1.)
    transform =  norm * np.array([
        [-g1-1., -g2],
        [-g2, g1-1.]
    ])

    return transform

def _transform_source2gal(pars):
    '''
    Rotation by intrinsic angle

    pars is a dict
    (x,y) is position on source plane
    '''

    theta_int = pars['theta_int']

    # want to 'subtract' orientation
    theta = -theta_int

    c, s = np.cos(theta), np.sin(theta)

    transform =  np.array([
        [c, -s],
        [s,  c]
    ])

    return transform

def _transform_gal2source(pars):
    '''
    Rotation by intrinsic angle

    pars is a dict
    (x,y) is position on source plane
    '''

    theta_int = pars['theta_int']

    c, s = np.cos(theta_int), np.sin(theta_int)

    transform =  np.array([
        [c, -s],
        [s,  c]
    ])

    return transform

def _transform_gal2disk(pars):
    '''
    Account for inclination angle
    pars is a dict
    (x,y) is position on galaxy plane
    '''

    sini = pars['sini']
    i = np.arcsin(sini)

    transform =  np.array([
        [1., 0],
        [0, 1. / np.cos(i)]
    ])

    return transform

def _transform_disk2gal(pars):
    '''
    Account for inclination angle
    pars is a dict
    (x,y) is position on galaxy plane
    '''

    sini = pars['sini']
    i = np.arcsin(sini)

    transform =  np.array([
        [1., 0],
        [0, np.cos(i)]
    ])

    return transform

def _obs2source(pars, x, y):
    '''
    pars is a dict
    (x,y) is position on obs plane

    returns: (x', y') in source plane
    '''

    transform = _transform_obs2source(pars)

    return _multiply(transform, x, y)

def _source2gal(pars, x, y):
    '''
    pars is a dict
    (x,y) is position on source plane

    returns: (x', y') in gal plane
    '''

    transform = _transform_source2gal(pars)

    return _multiply(transform, x, y)

def _gal2disk(pars, x, y):
    '''
    pars is a dict
    (x,y) is position on gal plane

    returns: (x', y') in disk plane
    '''

    transform = _transform_gal2disk(pars)

    return _multiply(transform, x, y)

def _source2obs(pars, x, y):
    '''
    pars is a dict
    (x,y) is position on source plane

    returns: (x', y') in obs plane
    '''

    transform = _transform_source2obs(pars)

    # Old way:
    # obs2source = _transform_obs2source(pars)
    # transform = np.linalg.inv(obs2source)

    return _multiply(transform, x, y)

def _gal2source(pars, x, y):
    '''
    pars is a dict
    (x,y) is position on gal plane

    returns: (x', y') in source plane
    '''

    transform = _transform_gal2source(pars)

    # Old way:
    # source2gal = _transform_source2gal(pars)
    # transform = np.linalg.inv(source2gal)

    return _multiply(transform, x, y)

def _disk2gal(pars, x, y):
    '''
    pars is a dict
    (x,y) is position on disk plane

    returns: (x', y') in gal plane
    '''

    transform = _transform_disk2gal(pars)

    # Old way:
    # gal2disk = _transform_gal2disk(pars)
    # transform = np.linalg.inv(gal2disk)

    return _multiply(transform, x, y)

def _eval_in_obs_plane(pars, x, y, speed=False):
    '''
    pars is a dict
    (x,y) is position on obs plane

    will eval speed map instead of velocity if speed is True
    '''

    xp, yp = _obs2source(pars, x, y)

    return _eval_in_source_plane(pars, xp, yp, speed=speed)

def _eval_in_source_plane(pars, x, y, speed=False):
    '''
    pars is a dict
    (x,y) is position on source plane

    will eval speed map instead of velocity if speed is True
    '''

    xp, yp = _source2gal(pars, x, y)

    return _eval_in_gal_plane(pars, xp, yp, speed=speed)

def _eval_in_gal_plane(pars, x, y, speed=False):
    '''
    pars is a dict
    (x,y) is position on galaxy plane

    will eval speed map instead of velocity if speed is True
    '''

    xp, yp = _gal2disk(pars, x, y)
    speed_map = _eval_in_disk_plane(pars, xp, yp, speed=True)

    if speed is True:

        return speed_map

    else:
        # euler angles which handle the vector aspect of velocity transform
        sini = pars['sini']
        phi = np.arctan2(yp, xp)

        return sini * np.cos(phi) * speed_map

def _eval_in_disk_plane(pars, x, y, speed=False):
    '''
    Evaluates model at posiiton array in the galaxy plane, where
    pos=(x,y) is defined relative to galaxy center

    pars is a dict with model parameters

    will eval speed map instead of velocity if speed is True

    # TODO: For now, this only works for the default model.
    We can make this flexible with a passed model name & builder,
    but not necessary yet & causes problems with numba version
    '''

    if speed is False:
        # Velocity is 0 in the z-hat direction for a disk galaxy
        return np.zeros(np.shape(x))

    r = np.sqrt(x**2 + y**2)

    atan_r = np.arctan(r  / pars['rscale'])

    v_r = pars['v0'] + (2./ np.pi) * pars['vcirc'] * atan_r

    return v_r
