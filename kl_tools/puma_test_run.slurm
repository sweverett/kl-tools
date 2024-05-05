#!/bin/bash
#SBATCH --job-name=klfiber
#SBATCH --output=/xdisk/timeifler/jiachuanxu/kl_fiber/outputs/KL_fiber-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --array=1-3
#SBATCH --cpus-per-task=1

### >>> High priority purchase-in time
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
###SBATCH --partition=standard
###SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

module load gsl/2.6
module load openmpi3/3.1.4
module load anaconda
conda init bash
source ~/.bashrc

conda activate kltools

# 2 fiber configurations x 12 SNR bins x 4 hlr x 10 sinis; 960
DATADIR=/xdisk/timeifler/jiachuanxu/kl_fiber
NSTEPS=50000
TNOM=600
SCRIPT=test_fiber_mcmc_run.py
NCPUS=${SLURM_NTASKS}

cd $SLURM_SUBMIT_DIR

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    mpirun -n ${NCPUS} --mca btl tcp,self --oversubscribe python ${SCRIPT} ${NSTEPS} -run_name=test_Ha_r_margffnorm -Iflux=3 -sini=8 -hlr=1 -PA=0 -fiberconf=0 -EXP_OFFSET=${TNOM} -EXP_PHOTO=4000 -PHOT_MASK=4 -SPEC_MASK=1 --mpi
fi

if [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    mpirun -n ${NCPUS} --mca btl tcp,self --oversubscribe python ${SCRIPT} ${NSTEPS} -run_name=test_Ha_margffnorm -Iflux=3 -sini=8 -hlr=1 -PA=0 -fiberconf=0 -EXP_OFFSET=${TNOM} -EXP_PHOTO=4000 -PHOT_MASK=0 -SPEC_MASK=1 --mpi
fi

if [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    mpirun -n ${NCPUS} --mca btl tcp,self --oversubscribe python ${SCRIPT} ${NSTEPS} -run_name=test_r_margffnorm -Iflux=3 -sini=8 -hlr=1 -PA=0 -fiberconf=0 -EXP_OFFSET=${TNOM} -EXP_PHOTO=4000 -PHOT_MASK=4 -SPEC_MASK=0 --mpi
fi
