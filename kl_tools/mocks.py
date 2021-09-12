import numpy as np
import sys
import os
import time
import logging
import galsim
from multiprocessing import Pool
import pudb

import utils
import cube

class MockSource(object):
    pass

def make_mock_chromatic_observations(config):
    '''
    Make mock observations of bulge+disk galaxies with specified
    SED and noise properties

    TODO: For now, just generating a single galaxy at given z, params, etc.
          The existing code has lots of similarity with GalSim Demo 12
    '''

    logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger('mock-chromatic-observations')

    if 'seed' in config:
        seed = config['seed']
    else:
        seed = 123456789

    if 'z' in config:
        redshift = config['z']
    else:
        redshift = 0.8

    logger.info(f'Using seed {seed}')
    logger.info(f'Using redshift {redshift}')

    # For now, simple achromatic PSF model
    fwhm = config['psf']['fwhm']
    beta = config['psf']['beta']
    psf = galsim.Moffat(fwhm=fwhm, beta=beta)

    # For now, single galaxy properties

    hlr_dv = config['dv']['hlr']
    hlr_exp = config['exp']['hlr']

    g1_dv = config['dv']['g1']
    g2_dv = config['dv']['g2']
    g1_exp = config['exp']['g1']
    g2_exp = config['exp']['g2']

    logger.info('')
    logger.info('Simulating chromatic bulge+disk galaxy')
    logger.info('')

    # make a bulge ...
    mono_bulge = galsim.DeVaucouleurs(half_light_radius=hlr_dv)
    bulge_SED = SEDs['CWW_E_ext'].atRedshift(redshift)
    bulge = mono_bulge * bulge_SED
    bulge = bulge.shear(g1=g1_dv, g2=g2_dv)
    logger.info('Created bulge component')

    # ... and a disk ...
    mono_disk = galsim.Exponential(half_light_radius=hlr_exp) # was 2 in demo
    disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
    disk = mono_disk * disk_SED
    disk = disk.shear(g1=g1_exp, g2=g2_exp)
    logger.info('Created disk component')

    # ... and then combine them.
    fracdev = config['fracdev']
    bdgal = fracdev*bulge + (1-fracdev)*disk
    bdfinal = galsim.Convolve([bdgal, psf])
    # Note that at this stage, our galaxy is chromatic but our PSF is still achromatic.  Part C)
    # below will dive into chromatic PSFs.
    logger.info('Created bulge+disk galaxy final profile')

    # Make image
    pixel_scale = config['image']['pixel_scale']
    box_size = config['image']['box_size']

    img = galsim.ImageF(box_size, box_size, scale=pixel_scale)
    bdfinal.drawImage(image=img)
    noise = galsim.GaussianNoise(rng, sigma=0.02)
    img.addNoise(noise)

    # ...

    return

def make_mock_COSMOS_observations(config):
    '''
    NOTE: "Users who wish to simulate F814W images with a different telescope
    and an exposure time longer than 1 second should multiply by that exposure
    time, and by the square of the ratio of the effective diameter of their
    telescope compared to that of HST.

    TODO: Do we need to worry about zeropoint offsets? COSMOS assums 25.94
    '''

    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
    logger = logging.getLogger('mock-cosmos-observations')

    outdir = config['outdir']
    utils.make_dir(outdir)

    outfile = os.path.join(outdir, 'kl-mocks-COSMOS.fits')

    # real_galaxy_catalog = galsim.RealGalaxyCatalog(config['cosmos_file'],
    #                                                dir=config['cosmos_dir'])
    use_real = config['use_real']
    area = config['telescope_area']
    exp_time = config['exposure_time']
    try:
        min_flux = config['min_flux']
    except KeyError:
        min_flux = 0
    cosmos_catalog = galsim.COSMOSCatalog(file_name=config['cosmos_file'],
                                          dir=config['cosmos_dir'],
                                          use_real=use_real,
                                          area=area,
                                          exptime=exp_time)
                                          # sample='25.2')
    # config['cosmos_catalog'] = cosmos_catalog

    Nobjs = cosmos_catalog.nobjects
    logger.info(f'Read in {Nobjs} real galaxies from catalog')

    # Determine number of bands per source
    bandpass_list = config['bandpass_list']
    Nspec = len(bandpass_list)

    # Make cutouts
    Ngals = config['ngals']
    logger.info(f'Starting generation of {Ngals} cutouts, ' +\
                f'each with {Nspec} bands')
    # im_list = []
    ncores = config['ncores']

    chromatic = config['chromatic']

    # for mp over bandpasses
    for i in range(Ngals):
        im_list = []

        # random selection
        gal = cosmos_catalog.makeGalaxy(chromatic=chromatic)

        # Don't understand this. makeGalaxy() returns a GSObject or
        # ChromaticObject, but when saved to gal it seems to make a
        # tuple for unknown reasons
        if type(gal) is tuple:
            assert len(gal) == 1
            gal = gal[0]

        # Want a consistent rotation angle between bands
        seed = config['seed']
        rng = galsim.UniformDeviate(seed+i+1)
        theta = 2.*np.pi * rng() * galsim.radians

        if ncores == 1:
            for k in range(Nspec):
                im_list.append(_make_single_COSMOS_im(
                    k, gal, bandpass_list[k], config, logger, theta=theta
                    ))

        else:
            with Pool(ncores) as pool:
                im_list.append(pool.starmap(_make_single_COSMOS_im,
                                            [(k,
                                              gal,
                                              bandpass_list[k],
                                              config,
                                              logger,
                                              theta)
                                             for k in range(Nspec)
                                             ]
                                            )
                            )

            # haven't figured out why it does this yet
            im_list = im_list[0]

        logger.info(f'Done making images for galaxy {i}')

        # Write the images to a fits data cube.
        outnum = str(i+1).zfill(3)
        im_outfile = outfile.replace('.fits', f'-{outnum}.fits')
        galsim.fits.writeCube(im_list, im_outfile)
        logger.info(f'Wrote image to fits data cube {im_outfile}')

    # for mp over gals
    # with Pool(ncores) as pool:
    #     im_list.append(pool.starmap(_make_single_COSMOS_im,
    #                                 [(k,
    #                                   cosmos_catalog.makeGalaxy(), # random selection
    #                                   config,
    #                                   logger)
    #                                  for k in range(Ngals)])
    #                    )

    # haven't figured out why it does this yet
    # im_list = im_list[0]

    logger.info('Done making all images')

    # Write the images to a fits data cube.
    # galsim.fits.writeCube(im_list, outfile)
    # logger.info(f'Wrote image to fits data cube {outfile}')

    return

def _make_single_COSMOS_im(k, gal, bandpass, config, logger, theta=None):
    logger.info(f'Start work on image {k}')
    t1 = time.time()

    # For now, make an achromatic double Gaussian PSF
    psf_inner_fwhm = config['psf_inner_fwhm']
    psf_outer_fwhm = config['psf_outer_fwhm']
    psf_inner_fraction = config['psf_inner_fraction']
    psf_outer_fraction = config['psf_outer_fraction']
    psf1 = galsim.Gaussian(fwhm=psf_inner_fwhm, flux=psf_inner_fraction)
    psf2 = galsim.Gaussian(fwhm=psf_outer_fwhm, flux=psf_outer_fraction)
    psf = psf1 + psf2

    # Draw the PSF with no noise.
    pixel_scale = config['pixel_scale']
    psf_image = psf.drawImage(scale=pixel_scale)

    # Initialize the random number generator we will be using.
    seed = config['seed']
    rng = galsim.UniformDeviate(seed+k+1)

    # gal = cosmos_catalog.makeGalaxy()
    # gal = galsim.RealGalaxy(real_catalog,
    #                         random=True,
    #                         flux_rescale=rescale)

    logger.info('   Read in training sample galaxy and PSF from file')
    t2 = time.time()

    # TODO: Rescale flux!
    # NOTE: Now trying this out through init of COSMOS catalog
    # rescale = config['flux_rescale']
    # gal *= rescale

    # Rotate by a random angle (but consistent between exp's)
    if theta is None:
        theta = 2.*np.pi * rng() * galsim.radians
    gal = gal.rotate(theta)

    # Apply the desired shear
    shear_g1, shear_g2 = config['shear_g1'], config['shear_g2']
    gal = gal.shear(g1=shear_g1, g2=shear_g2)

    # Same for magnification
    mu = config['mu']
    gal = gal.magnify(mu)

    # Make the combined profile
    final = galsim.Convolve([psf, gal])

    # Offset centroid
    max_offset = config['max_pixel_offset']
    dx = rng() - max_offset
    dy = rng() - max_offset

    # Draw the profile
    box_size = config['box_size']
    nx, ny = box_size, box_size

    if bandpass is None:
        im = final.drawImage(scale=pixel_scale, nx=nx, ny=ny, offset=(dx, dy))
    else:
        im = final.drawImage(bandpass, scale=pixel_scale, nx=nx, ny=ny, offset=(dx, dy))

    logger.info('   Drew image')
    t3 = time.time()

    # Add Gaussian noise
    noise_sigma = config['noise_sigma']
    im.addNoise(galsim.GaussianNoise(rng=rng,
                                     sigma=noise_sigma))

    # # Add a constant background level
    # sky_level = config['sky_level']
    # background = sky_level * pixel_scale**2
    # im += background

    # Add noise
    # im.addNoise(galsim.CCDNoise(sky_level=sky_level,
    #                             )
    # Add Poisson noise
    # im.addNoise(galsim.PoissonNoise(rng))

    # logger.info('   Added Poisson noise')
    t4 = time.time()

    return im

def make_test_COSMOS_config():
    config = {
        # COSMOS params
        'cosmos_dir': '/Users/sweveret/miniconda3/envs/sbmcal/share/' + \
                      'galsim/COSMOS_23.5_training_sample/',
        'cosmos_file': 'real_galaxy_catalog_23.5.fits',
        'outdir': os.path.join(utils.TEST_DIR,
                               'mocks',
                               'COSMOS'),
        'use_real': False,
        'chromatic': True,
        # Telescope params
        'flux_rescale': 1.0, # Should be replaced by specific telescope info
        # Sim params
        'seed': 1512413,
        'zeropoint': 25.94, # This one is for COSMOS
        'sky_level': 0,       # ADU / arcsec^2
        'max_pixel_offset': 0.05, # pixels
        'pixel_scale': 0.5,      # arcsec / pixel
        'shear_g1': -0.027,
        'shear_g2': 0.031,
        'mu': 1.082,
        'psf_inner_fwhm': 0.6,    # arcsec
        'psf_outer_fwhm': 2.3,    # arcsec
        'psf_inner_fraction': 0.8,  # fraction of PSF flux in the inner Gaussian
        'psf_outer_fraction': 0.2,  # fraction of PSF flux in the outer Gaussian
        'noise_sigma': 50, # counts
        'box_size': 32,
        'ngals': 10,
        'ncores': 8,
        # params related to SED
        'throughput': '0.85', # str that can be evaluated as a function
        'throughput_unit': 'nm', # Unit for 'wave' in throughput func
        'lambda_start': 500, # nm
        'lambda_end': 1000, # nm
        'dlambda': 50, # nm
        }

    return config

def make_test_chromatic_config():
    config = {
        'outdir': os.path.join(utils.TEST_DIR,
                               'mocks'),
        'z': 0.8,
        'psf': {
            'fwhm': 0.7, # arcsec
            'beta': 2.5
        },
        'exp': {
            'g1': 0.4,
            'g2': 0.2,
            'hlr': 2.
        },
        'dv': {
            'g1': .12,
            'g2': .07,
            'hlr': 0.5
        },
        'fracdev': 0.5,
        'image': {
            'pix_scale': 0.2,
            'box_size': 64
        }
    }

    return config

def make_bandpass_list(config):
    '''
    Make a list of bandpasses given throughput function
    and wavelength limits in config. For now, assumes a
    constant dlambda
    '''

    throughput = config['throughput']
    unit = config['throughput_unit']
    li, le = config['lambda_start'], config['lambda_end']
    dl = config['dlambda']
    zp = config['zeropoint']

    bandpass_list = []

    for l in range(li, le+dl, dl):
        bandpass_list.append(galsim.Bandpass(
            throughput, unit, blue_limit=li, red_limit=le, zeropoint=zp
            ))

    return bandpass_list

def main():

    config = make_test_COSMOS_config()

    # The catalog returns objects that are appropriate for HST in 1 second exposures.  So for another
    # telescope we scale up by the relative area and exposure time.  Note that what is important is
    # the *effective* area after taking into account obscuration.  For HST, the telescope diameter
    # is 2.4 but there is obscuration (a linear factor of 0.33).  Here, we assume that the telescope
    # we're simulating effectively has no obscuration factor.  We're also ignoring the pi/4 factor
    # since it appears in the numerator and denominator, so we use area = diam^2.
    tel_diam = 5.1 # m
    exp_time = 90 # s
    hst_eff_area  = 2.4**2 * (1.-0.33**2)
    hale_eff_area = 20 # m^2
    flux_rescale = (tel_diam**2/hst_eff_area) * exp_time
    config['flux_rescale'] = flux_rescale

    config['telescope_area'] = hale_eff_area * (100.)**2 # cm
    config['exposure_time'] = exp_time

    config['min_flux'] = 100

    # Setup bandpasses
    config['bandpass_list'] = make_bandpass_list(config)

    make_mock_COSMOS_observations(config)

    return 0

if __name__ == '__main__':
    rc = main()

    if rc == 0:
        print('Tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
