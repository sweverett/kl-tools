import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

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

To make numba work, it requires making some assumptions about the structure of
the pars list. This is a dict in no_numba_transformation.py to be flexible on
models.
'''

@njit
def transform_coords(x, y, plane1, plane2, pars):
    '''
    Transform coords (x,y) defined in plane1 into plane2

    pars: array holding model information
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

@njit
def _multiply(transform, pos):
    '''
    transform: a (2x2) coordinate transformation matrix
    pos: np.ndarray [x, y]
        x: np.array of x position
        y: np.array of y position

    returns: np.ndarray [x', y']
    '''

    # Can't just do a matrix multiplication because we can't assume
    # structure of pos; e.g. it might be a meshgrid array

    x = pos[0]
    y = pos[1]

    assert x.shape == y.shape

    new_pos = np.empty(pos.shape)

    # x,y are vectors
    if len(x.shape) == 1:
        # have to do it this way for numba
        pos = np.empty((2, len(x)))
        pos[0] = x
        pos[1] = y

        # numba doesn't seem to work with matmul
        # out = np.matmul(transform, pos)
        # new_pos[0,1] = np.dot(transform, pos)
        # xp, yp = out[0], out[1]

    # x,y are matrices
    elif len(x.shape) == 2:
        # gotta do it the slow way
        # out = np.zeros((2, x.shape[0], x.shape[1]))
        # xp = np.empty(x.shape)
        # yp = np.empty(y.shape)

        new_pos = np.zeros(pos.shape)

        N1, N2 = x.shape
        for i in range(N1):
            # have to do it this way for numba
            # pos = np.array([x[i,:], y[i,:]])
            # pos = np.empty((2, len(x)))
            # pos[0] = x[i,:]
            # pos[1] = y[i,:]

            pos_slice = pos[i,:]

            # numba doesn't seem to work with matmul
            # out = np.matmul(transform, pos)
            new_pos[i,:] = np.dot(transform, pos_slice)
            # xp[i,:] = out[0]
            # yp[i,:] = out[1]

    else:
        raise ValueError('Plane transformations are only implemented ' +\
                         'for scalars, vectors, and arrays!')

    # return xp, yp
    return new_pos

# The following transformation definitions require basic knowledge
# of the source shear, intrinsic orientation, and profile inclination
# angle
#
# The inverse transforms are computed on the fly from these defs
# TODO: Could try to work out these inv transforms analytically
#       for faster eval

@njit
def _transform_obs2source(pars):
    '''
    Lensing transformation from obs to source plane

    pars is an array
    (x,y) is position on obs plane
    '''

    # PARS_DEF = get_pars_def()

    # g1 = pars[PARS_DEF['g1']]
    # g2 = pars[PARS_DEF['g2']]

    # See PARS_DEF
    g1, g2 = pars[0], pars[1]

    # Lensing transformation
    transform =  np.array([
        [1.-g1, -g2],
        [-g2, 1.+g1]
    ])

    return transform

@njit
def _transform_source2gal(pars):
    '''
    Rotation by intrinsic angle

    pars is an array
    (x,y) is position on source plane
    '''

    # see PARS_DEF
    theta_int = pars[2]

    # want to 'subtract' orientation
    theta = -theta_int

    c, s = np.cos(theta), np.sin(theta)

    transform =  np.array([
        [c, -s],
        [s,  c]
    ])

    return transform

@njit
def _transform_gal2disk(pars):
    '''
    Account for inclination angle
    pars is an array
    (x,y) is position on galaxy plane
    '''

    # sini = pars[PARS_DEF['sini']]
    # see PARS_DEF
    sini = pars[3]
    i = np.arcsin(sini)

    transform =  np.array([
        [1., 0],
        [0, 1. / np.cos(i)]
    ])

    return transform

@njit
def _obs2source(pars, x, y):
    '''
    pars is a list
    (x,y) is position on obs plane

    returns: (x', y') in source plane
    '''

    transform = _transform_obs2source(pars)

    return _multiply(transform, x, y)

@njit
def _source2gal(pars, x, y):
    '''
    pars is an array
    (x,y) is position on source plane

    returns: (x', y') in gal plane
    '''

    transform = _transform_source2gal(pars)

    return _multiply(transform, x, y)

@njit
def _gal2disk(pars, x, y):
    '''
    pars is an array
    (x,y) is position on gal plane

    returns: (x', y') in disk plane
    '''

    transform = _transform_gal2disk(pars)

    return _multiply(transform, x, y)

@njit
def _source2obs(pars, x, y):
    '''
    pars is an array
    (x,y) is position on source plane

    returns: (x', y') in obs plane
    '''

    obs2source = _transform_obs2source(pars)

    transform = np.linalg.inv(obs2source)

    return _multiply(transform, x, y)

@njit
def _gal2source(pars, x, y):
    '''
    pars is an array
    (x,y) is position on gal plane

    returns: (x', y') in source plane
    '''

    source2gal = _transform_source2gal(pars)

    transform = np.linalg.inv(source2gal)

    return _multiply(transform, x, y)

@njit
def _disk2gal(pars, x, y):
    '''
    pars is an array
    (x,y) is position on disk plane

    returns: (x', y') in gal plane
    '''

    gal2disk = _transform_gal2disk(pars)

    transform = np.linalg.inv(gal2disk)

    return _multiply(transform, x, y)

@njit
def _eval_in_obs_plane(pars, x, y, speed=False):
    '''
    pars is an array
    (x,y) is position on obs plane

    will eval speed map instead of velocity if speed is True
    '''

    xp, yp = _obs2source(pars, x, y)

    return _eval_in_source_plane(pars, xp, yp, speed=speed)

@njit
def _eval_in_source_plane(pars, x, y, speed=False):
    '''
    pars is an array
    (x,y) is position on source plane

    will eval speed map instead of velocity if speed is True
    '''

    xp, yp = _source2gal(pars, x, y)

    return _eval_in_gal_plane(pars, xp, yp, speed=speed)

@njit
def _eval_in_gal_plane(pars, x, y, speed=False):
    '''
    pars is an array
    (x,y) is position on galaxy plane

    will eval speed map instead of velocity if speed is True
    '''

    xp, yp = _gal2disk(pars, x, y)
    speed_map = _eval_in_disk_plane(pars, xp, yp, speed=True)

    if speed is True:

        return speed_map

    else:
        # euler angles which handle the vector aspect of velocity transform
        sini = pars[3]
        phi = np.arctan2(yp, xp)

        return sini * np.cos(phi) * speed_map

@njit
def _eval_in_disk_plane(pars, x, y, speed=False):
    '''
    Evaluates model at posiiton array in the galaxy plane, where
    pos=(x,y) is defined relative to galaxy center

    pars is an array with model parameters

    will eval speed map instead of velocity if speed is True

    # TODO: For now, this only works for the default model.
    We can make this flexible with a passed model name & builder,
    but not necessary yet
    '''

    if speed is False:
        # Velocity is 0 in the z-hat direction for a disk galaxy
        return np.zeros(np.shape(x))

    r = np.sqrt(x**2 + y**2)

    # Grab relevant pars
    # See PARS_ORDER
    v0 = pars[4]
    vcirc = pars[5]
    rscale = pars[6]

    atan_r = np.arctan(r  / rscale)

    v_r = v0 + (2. / np.pi) * vcirc * atan_r

    return v_r

def main():
    print('Starting multiply test')
    x = np.random.randn(10, 10)
    y = np.random.randn(10, 10)
    pos = np.array([x, y])

    pudb.set_trace()

    transform = np.array([
        [1., 0.],
        [0., 1.]
        ])
    xp, yp = _multiply(transform, pos)

    print('Done!')

    return

if __name__ == '__main__':
    main()
