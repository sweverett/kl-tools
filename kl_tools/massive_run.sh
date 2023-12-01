#!/bin/bash

RUN=bgs_like_array
NSTEPS=2000
SCRIPT=test_fiber_mcmc_run.py
NCPUS=4
SINI=0.05
HLR=1.5
FIBERCONF=0

for (( c=0; c<30; c++ ))
do
    mpirun -n ${NCPUS} python ${SCRIPT} ${NSTEPS} -run_name=${RUN} -flux_scaling_power=${c}  -sini=${SINI} -hlr=${HLR} -fiber_conf=${FIBERCONF} --mpi
done

