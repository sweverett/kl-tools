#!/bin/bash

#SBATCH --job-name=KLnS_high
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/JWSTKL_noshear_highpri_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=1
### there are 69 sources in the high priority sample
#SBATCH --array=1-69

### >>> High priority purchase-in time
###SBATCH --partition=high_priority
###SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
#SBATCH --partition=standard
#SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load anaconda
module load openmpi5
module load gsl
conda init bash
source ~/.bashrc

cd /home/u17/jiachuanxu/kl-tools/scripts
conda activate kltools
#MPIRUN=/opt/ohpc/pub/mpi/openmpi5-gnu13/5.0.5/bin/mpirun
#MPIRUN=/opt/ohpc/pub/mpi/openmpi3-gnu8/3.1.4/bin/mpirun
MPIRUN=/groups/timeifler/jiachuanxu/python_envs/envs/kltools/bin/mpirun

### Real data (high priority sources)
YAML_LIST=(../yaml/high_priority/obj*_noshear_TFR_full_equalT.yaml)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
YAML="${YAML_LIST[$IDX]}"
echo "Running with file $YAML"

### pocoMC config
N_EFFECTIVE=512
N_TOTAL=4096
#N_EFFECTIVE=2048
#N_TOTAL=16384

MPI_PML="--mca pml ob1"
MPI_BTL="--mca btl tcp,self"

### emcee
#### ${MPIRUN} -n ${SLURM_NTASKS} ${MPI_BTL} python test_jwst_mcmc_run.py ${YAML} --mpi

### pocoMC
${MPIRUN} -n ${SLURM_NTASKS} python test_jwst_mcmc_run.py ${YAML} -sampler=poco -ID=-1 -nparticles=${N_EFFECTIVE} -n_total=${N_TOTAL} --mpi

### ultranest
#mpirun -n ${SLURM_NTASKS} python test_jwst_mcmc_run.py ${YAML} -sampler=ultranest -nparticles=${N_EFFECTIVE} -n_total=100 --mpi

