#!/bin/bash
#SBATCH --job-name=klfiber
#SBATCH --output=/xdisk/timeifler/jiachuanxu/kl_fiber/outputs/KL_fiber-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --array=1-500
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

module load gsl/2.6
module load openmpi3/3.1.4
module load anaconda
conda init bash
source ~/.bashrc

conda activate kltools

# 2 fiber configurations x 12 SNR bins x 4 hlr x 10 sinis; 960
DATADIR=/xdisk/timeifler/jiachuanxu/kl_fiber
RUN=SNR-HDR_multiline_3image_tnom600_rmaglim
NSTEPS=10000
TNOM=600
SCRIPT=test_fiber_mcmc_run.py
NCPUS=${SLURM_NTASKS}
hit=0

cd $SLURM_SUBMIT_DIR
# fiber configuration
#for (( a=0; a<2; a++ ))
#for a in 0 3
#do

# flux 
for (( b=0; b<10; b++ ))
do
	# sini
	for (( a=0; a<10; a++ ))
	do 
		# rmag limit
		for c in 60 360 4000 13000 40000
		do
			# SLURM arrays start from 1
			hit=$((${hit}+1))
			if [ ${hit} -eq ${SLURM_ARRAY_TASK_ID} ]
			then
				# run chains
				mpirun -n ${NCPUS} --mca btl tcp,self --oversubscribe python ${SCRIPT} ${NSTEPS} -run_name=${RUN} -Iflux=${b} -sini=${a} -hlr=1 -fiberconf=0 -sigma_int=1 -PA=0 -EXP_OFFSET=${TNOM} -EXP_PHOTO=${c} -NPHOT=1 --mpi
			fi
		done
	done
done
