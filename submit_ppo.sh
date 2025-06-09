#!/bin/bash
#SBATCH -J ModularRodentImitation --ntasks=1 --cpus-per-task 16 --mem-per-cpu 1000 -p olveczkygpu -t 1-00:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu

module add intel
module add openmpi
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export JULIA_CPU_TARGET="sapphirerapids;skylake-avx512;cascadelake;icelake-server"

#srun --mpi=pmix -n 1 : -n 15\
julia --threads=16 --project=. src/test_modular.jl configs/modular_imitation.toml -wandb.run_name "ModularImitation-{NOW}"


