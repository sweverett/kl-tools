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

#SBATCH --time=10:00:00
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

### Real data
#YAML=../yaml/example_jwst_noshear_freekine.yaml
#YAML=../yaml/example_jwst_shear_TFR_short.yaml
#YAML=../yaml/example_jwst_noshear_TFR_short.yaml
YAML=../yaml/example_jwst_shear_TFR_full_D2.yaml

### Mock data
#YAML=../yaml/mock_jwst_shear_TFR_smooth8.yaml
#YAML=../yaml/mock_jwst_shear_TFR.yaml
#YAML=../yaml/mock_jwst_shear_TFR_full.yaml

### pocoMC config
#N_EFFECTIVE=512
#N_TOTAL=4096
N_EFFECTIVE=2048
N_TOTAL=16384

MPI_PML="--mca pml ob1"
MPI_BTL="--mca btl tcp,self"

### emcee
#### ${MPIRUN} -n ${SLURM_NTASKS} ${MPI_BTL} python test_jwst_mcmc_run.py ${YAML} --mpi

### pocoMC
${MPIRUN} -n ${SLURM_NTASKS} ${MPI_BTL} python test_jwst_mcmc_run.py ${YAML} --mpi -sampler=pocomc -nparticles=${N_EFFECTIVE} -n_total=${N_TOTAL} --mpi


