#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task 16 --mem-per-cpu 500 -p olveczkygpu -t 0-06:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu

module add intel
module add openmpi
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export JULIA_CPU_TARGET="sapphirerapids;skylake-avx512;cascadelake;icelake-server"
srun --mpi=pmix -n 1 julia --threads=16 --project=. experiments/train_joystick.jl

