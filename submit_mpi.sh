#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task 32 --mem-per-cpu 1000 -p gpu -t 1-00:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu
#SBATCH hetjob
#SBATCH --nodes=7 --ntasks-per-node=1 --cpus-per-task 32 --mem-per-cpu 1000 -p sapphire -t 1-00:00 -o %j.out -e %j.err

module add intel
module add openmpi
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
#unset LD_LIBRARY_PATH
srun --mpi=pmix -n 1 : -n 7 julia --threads=32 --project=. src/run_ppo.jl #--het-group=0,1

