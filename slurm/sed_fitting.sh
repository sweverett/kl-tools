#!/bin/bash

#SBATCH --job-name=SEDFIT
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/JWST_SED-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

### >>> High priority purchase-in time
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
###SBATCH --partition=standard
###SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=72:00:00
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

CATALOG="/xdisk/timeifler/jiachuanxu/jwst/fengwu_catalog/v094_matched_phot_JADE_LS.fits"

python stellar_mass_sed_fitting.py --add_neb --add_duste --phottable=${CATALOG} --ind=${SLURM_ARRAY_TASK_ID} --nlive_init=400 --nested_target_n_effective=1000 --nested_dlogz_init=0.05 --dynesty

