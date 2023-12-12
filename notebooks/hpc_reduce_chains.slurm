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

# 2 fiber configurations x 12 SNR bins x 4 hlr x 10 sinis; 960
RUN=bgs_like_array_tnom600
NSTEPS=5000
SCRIPT=test_fiber_mcmc_run.py
NCPUS=${SLURM_NTASKS}
hit=0
# fiber configuration
for (( a=0; a<2; a++ ))
do
	# flux 
	for (( b=0; b<12; b++ ))
	do
		# hlr 
		for (( c=0; c<4; c++ ))
		do
			# sini
			for (( d=0; d<10; d++ ))
			do
				# SLURM arrays start from 1
				hit=$((${hit}+1))
				if [ ${hit} -eq ${SLURM_ARRAY_TASK_ID} ]
				then
					# reduce chains
					python reduce_chains.py -run_name=${RUN} -Iflux=${b}  -sini=${d} -hlr=${c} -fiberconf=${a}
				fi
			done
		done
	done
done
