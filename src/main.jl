import MPI
import BSON
import CUDA
import Dates
import Statistics
import HDF5
import LinearAlgebra: norm
import MuJoCo
import PythonCall
import Wandb
import Random
import TOML
using Flux
using ProgressMeter
using StaticArrays

include("utils/profiler.jl")
include("utils/load_dm_control_model.jl")
include("utils/mujoco_quat.jl")
include("utils/component_tensor.jl")
include("utils/wandb_logger.jl")
include("utils/parse_config.jl")
include("environments/mujoco_env.jl")
include("environments/imitation_trajectory.jl")
include("environments/rodent_imitation_env.jl")
include("collectors/batch_stepper.jl")
include("collectors/mpi_stepper.jl")
include("collectors/cuda_collector.jl")
include("algorithms/ppo.jl")
include("networks/utils.jl")
include("networks/variational_bottleneck.jl")
include("networks/info_bottleneck.jl")
include("networks/action_samplers.jl")
include("networks/enc_dec.jl")

params = parse_config(ARGS)
env = RodentImitationEnv(params) #There is just one supported env
ppo(env, params; use_mpi=params.rollout.use_mpi)
