import Dates
import TOML
using BenchmarkTools

include("../src/environments/environments.jl")

walker = Environments.ModularRodent(min_torso_z=0.035, spawn_z_offset=0.01, n_physics_steps=5)

Environments.proprioception(walker)
precomputed_target = Environments.precompute_target(walker)

Environments.ModularImitationEnv(walker; max_target_distance=0.1)

