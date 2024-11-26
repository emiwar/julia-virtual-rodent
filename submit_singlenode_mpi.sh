#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task 20 --mem-per-cpu 1000 -p olveczkygpu -t 0-96:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu

module add intel
module add openmpi
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export JULIA_CPU_TARGET="sapphirerapids;skylake-avx512;cascadelake;icelake-server"
srun --mpi=pmix -n 1 julia --threads=20 --project=. src/run_ppo.jl
