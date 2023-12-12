#!/bin/bash
#SBATCH --job-name=klfiber
#SBATCH --output=KL_fiber-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-960
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

### >>> High priority purchase-in time
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
###SBATCH --partition=standard
###SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

module load gsl/2.6
module load openmpi3/3.1.4
module load anaconda
conda init bash
source ~/.bashrc

cd $SLURM_SUBMIT_DIR
conda activate kltools

python reduce_chains.py ${SLURM_ARRAY_TASK_ID}
