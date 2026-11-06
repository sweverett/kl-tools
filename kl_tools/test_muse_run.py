# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys
import pickle
import schwimmbad
import mpi4py
from schwimmbad import MPIPool
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
import zeus

import utils
from mcmc import KLensZeusRunner, KLensEmceeRunner
import priors
import cube
import likelihood
from parameters import PARS_ORDER
from likelihood import log_posterior
from velocity import VelocityMap

import pudb

parser = ArgumentParser()

parser.add_argument('nsteps', type=int,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('-run_name', type=str, default='',
                    help='Name of mcmc run')
parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing`)')
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')
parser.add_argument("--datacube",type=str,"path to 3d spectrum")
parser.add_argument("--spec1d",type=str,help="path to 1d spectrum, needed for wavelength calibration")
parser.add_argument("--catalog",type=str,help="path to catalog file, needed for isolating lines.")


def main(args, pool):
    nsteps = args.nsteps
    sampler = args.sampler
    ncores = args.ncores
    mpi = args.mpi
    run_name = args.run_name
    show = args.show
    
    outdir = os.path.join(utils.TEST_DIR, 'test-mcmc-run', run_name)
    utils.make_dir(outdir)

    # Looks like we can ignore true_pars, as set in test_mcmc_run.py, as long as we don't need to run the tests.

    halpha = 656.28 # nm
    R = 3000. # This is a guess, based on this: https://www.eso.org/sci/facilities/paranal/instruments/muse/inst.html

    
    
