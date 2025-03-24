#!/bin/bash

#SBATCH --job-name=JWSTKL
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/JWSTKL-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=1

### >>> High priority purchase-in time
###SBATCH --partition=high_priority
###SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
#SBATCH --partition=standard
#SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load anaconda
module load openmpi3
module load gsl
conda init bash
source ~/.bashrc

cd /home/u17/jiachuanxu/kl-tools/scripts
conda activate kltools
#MPIRUN=/opt/ohpc/pub/mpi/openmpi5-gnu13/5.0.5/bin/mpirun
MPIRUN=/opt/ohpc/pub/mpi/openmpi3-gnu8/3.1.4/bin/mpirun

YAML=../yaml/example_jwst_noshear_freekine.yaml

MPI_PML="--mca pml ob1"
MPI_BTL="--mca btl tcp,self"
${MPIRUN} -n ${SLURM_NTASKS} ${MPI_BTL} python test_jwst_mcmc_run.py ${YAML} --mpi



