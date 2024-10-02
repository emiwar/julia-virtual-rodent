#!/bin/bash
#SBATCG -J rodentImitation
#SBATCH -c 24                # Number of cores (-c)
#SBATCH -t 0-48:00          # Runtime in D-HH:MM, minimum of 5 minutes
#SBATCH -p olveczkygpu     # Partition to submit to
#SBATCH --mem=20000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o slurm_logs/%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm_logs/%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu

julia --threads 24 --project=. src/run_ppo.jl
