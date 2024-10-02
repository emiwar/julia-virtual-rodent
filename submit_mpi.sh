#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task 32 --mem-per-cpu 1000 -p gpu -t 0-00:15 -o %j_gpu.out -e %j_gpu.err --gres=gpu
#SBATCH hetjob
#SBATCH --ntasks=7 --cpus-per-task 32 --mem-per-cpu 1000 -p sapphire -t 0-00:15 -o %j.out -e %j.err

module add intel
module add openmpi
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
#unset LD_LIBRARY_PATH
srun --mpi=pmix -n 1 : -n 7 julia --threads=32 --project=. src/run_ppo.jl #--het-group=0,1

