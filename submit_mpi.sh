#!/bin/bash
#SBATCH -J rodentImitation
#SBATCH --ntasks=1 --cpus-per-task 32 --mem-per-cpu 1000 -p gpu -t 3-00:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu
#SBATCH hetjob
#SBATCH --ntasks=15 --cpus-per-task 32 --mem-per-cpu 1000 -p sapphire,shared -t 3-00:00 -o %j.out -e %j.err

module add intel
module add openmpi
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export JULIA_CPU_TARGET="sapphirerapids;skylake-avx512;cascadelake;icelake-server"

srun --mpi=pmix -n 1 : -n 15\
     julia --threads=32 --project=. src/main.jl\
     configs/imitation_sota.toml\
     -wandb.run_name "Basic-MLP-again-{NOW}"\
     -network.decoder_type "MLP"
