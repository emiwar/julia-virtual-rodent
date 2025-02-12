#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task 1 --mem=24000 -p gpu -t 0-08:00 -o %j_gpu.out -e %j_gpu.err --gres=gpu

module add intel
module add openmpi
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export JULIA_CPU_TARGET="sapphirerapids;skylake-avx512;cascadelake;icelake-server"
#srun --mpi=pmix -n 1 julia --threads=16 --project=. src/analysis/rollout_and_log_activations.jl mm0dnyq4
#srun --mpi=pmix -n 1 julia --threads=16 --project=. src/analysis/rollout_and_log_activations.jl j0zwbgns
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl mm0dnyq4 art/2020_12_22_1
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl j0zwbgns art/2020_12_22_1
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl mm0dnyq4 art/2020_12_22_2
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl j0zwbgns art/2020_12_22_2
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl mm0dnyq4 art/2020_12_22_3
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl j0zwbgns art/2020_12_22_3
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl j0zwbgns art/2020_12_23_1
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl j0zwbgns bud/2021_06_21_1
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl j0zwbgns bud/2021_06_21_2
srun --mpi=pmix -n 1 julia --project=. src/analysis/rollout_and_log_activations.jl j0zwbgns bud/2021_06_22_1
