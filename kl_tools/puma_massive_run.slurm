#!/bin/bash
#SBATCH --job-name=klfiber
#SBATCH --output=KL_fiber-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --array=1-800
#SBATCH --cpus-per-task=1

### >>> High priority purchase-in time
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
###SBATCH --partition=standard
###SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=4:00:00
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
RUN=SNR-HDR_sharp_tnom900
NSTEPS=5000
TNOM=900
SCRIPT=test_fiber_mcmc_run.py
NCPUS=${SLURM_NTASKS}
hit=0

cd $SLURM_SUBMIT_DIR
# fiber configuration
#for (( a=0; a<2; a++ ))
for a in 0 3
do
	# flux 
	for (( b=0; b<10; b++ ))
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
                    for subit in 0 1 2
                    do
					    # run chains
					    mpirun -n ${NCPUS} --mca btl tcp,self --oversubscribe python ${SCRIPT} ${NSTEPS} -run_name=${RUN} -Iflux=$(( b + subit * 10)) -sini=${d} -hlr=${c} -fiberconf=${a} -EXPTIME=${TNOM} --mpi
					    # reduce chains
					    python ../notebooks/reduce_chains.py -run_name=${RUN} -Iflux=$(( b + subit * 10))  -sini=${d} -hlr=${c} -fiberconf=${a}
                    done

#					# run chains
#					mpirun -n ${NCPUS} --mca btl tcp,self --oversubscribe python ${SCRIPT} ${NSTEPS} -run_name=${RUN} -Iflux=${b}  -sini=${d} -hlr=${c} -fiberconf=$(( a + 2)) --mpi
#					# reduce chains
#					python ../notebooks/reduce_chains.py -run_name=${RUN} -Iflux=${b}  -sini=${d} -hlr=${c} -fiberconf=$(( a + 2 ))

				fi
			done
		done
	done
done
