#!/bin/bash
#SBATCH -J rodentImitation
#SBATCH --ntasks=1 --cpus-per-task 32 --mem-per-cpu 1000 -p gpu -t 3-00:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu --constraint="holyhdr|holyndr"
#SBATCH hetjob
#SBATCH --ntasks=7 --cpus-per-task 32 --mem-per-cpu 1000 -p olveczky_sapphire,sapphire,shared -t 3-00:00 -o %j.out -e %j.err --constraint="holyhdr|holyndr"

module add intel
module add openmpi
export JULIA_PKG_PRECOMPILE_AUTO=0 #Precompilation is annoying with MPI
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export JULIA_CPU_TARGET="sapphirerapids;skylake-avx512;cascadelake;icelake-server"

srun --mpi=pmix -n 1 : -n 7\
     julia --threads=32 --project=. src/main.jl\
     configs/imitation_sota.toml\
     -wandb.run_name "HalfSizeBenchmark-{NOW}"\
     -rollout.n_envs 1024
