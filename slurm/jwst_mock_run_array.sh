#!/bin/bash

#SBATCH --job-name=JWSTKL
#SBATCH --output=/xdisk/timeifler/jiachuanxu/kltools_jwst_chains/mock_jwst_analysis/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=1

### Puma
### >>> High priority purchase-in time
###SBATCH --partition=high_priority
###SBATCH --qos=user_qos_timeifler

### Ocelote
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

### chooise your MPI: Puma or Ocelote
### Ocelote
module load openmpi3
MPIRUN=/opt/ohpc/pub/mpi/openmpi3-gnu8/3.1.4/bin/mpirun
### Puma
#module load openmpi5
#MPIRUN=/opt/ohpc/pub/mpi/openmpi5-gnu13/5.0.5/bin/mpirun

conda init bash
source ~/.bashrc

cd /home/u17/jiachuanxu/kl-tools/scripts
conda activate kltools

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Loaded modules: $(module list)


YAML=../yaml/mock_jwst_shear_TFR_${SLURM_ARRAY_TASK_ID}.yaml

### pocoMC config
N_EFFECTIVE=512
N_TOTAL=4096
#N_EFFECTIVE=1024
#N_TOTAL=8192

#N_EFFECTIVE=2048
#N_TOTAL=16384

MPI_PML="--mca pml ob1"
MPI_BTL="--mca btl tcp,self"

### emcee
#### ${MPIRUN} -n ${SLURM_NTASKS} ${MPI_BTL} python test_jwst_mcmc_run.py ${YAML} --mpi

### pocoMC
${MPIRUN} -n ${SLURM_NTASKS} ${MPI_BTL} ${MPI_PML} python test_jwst_mcmc_run.py ${YAML} --mpi -sampler=pocomc -nparticles=${N_EFFECTIVE} -n_total=${N_TOTAL} --run_from_params


