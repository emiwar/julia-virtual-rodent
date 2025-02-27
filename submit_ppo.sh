#!/bin/bash
#SBATCH -J LSTMRodentImitation --ntasks=1 --cpus-per-task 28 --mem-per-cpu 1000 -p olveczkygpu -t 2-00:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu

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


