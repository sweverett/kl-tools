#!/bin/bash

#SBATCH --job-name=JWSTKL
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/JWSTKL-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1

### >>> High priority purchase-in time
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
###SBATCH --partition=standard
###SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=140:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load anaconda
module load openmpi3
module load gsl
conda init bash
source ~/.bashrc

cd $SLURM_SUBMIT_DIR
conda activate kltools

/opt/ohpc/pub/mpi/openmpi3-gnu8/3.1.4/bin/mpirun -n ${SLURM_NTASKS} --mca btl tcp,self python test_jwst_mcmc_run.py ../yaml/example_jwst.yaml --mpi



