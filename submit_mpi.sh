#!/bin/bash
#SBATCH -J rodentImitation
#SBATCH --ntasks=1 --cpus-per-task 32 --mem-per-cpu 1000 -p gpu -t 3-00:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu --constraint="holyhdr|holyndr" --no-requeue
#SBATCH hetjob
#SBATCH --ntasks=15 --cpus-per-task 32 --mem-per-cpu 1000 -p sapphire,shared -t 3-00:00 -o %j.out -e %j.err --constraint="holyhdr|holyndr" --no-requeue

module add intel
module add openmpi
export JULIA_PKG_PRECOMPILE_AUTO=0 #Precompilation is annoying with MPI
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export JULIA_CPU_TARGET="sapphirerapids;skylake-avx512;cascadelake;icelake-server"

srun --mpi=pmix -n 1 : -n 15\
     julia --threads=32 --project=. src/main.jl\
     configs/simplified_imitation.toml


