module Environments
import MuJoCo
import PythonCall
import HDF5
import LinearAlgebra: norm
using StaticArrays
using ..ComponentTensors

include("AbstractEnv.jl")

include("../utils/mujoco_quat.jl")

#Imitation
include("Walker.jl")
include("Rodent.jl")
include("imitation_utils.jl")
include("imitation_env.jl")
include("imitation_reward_spec.jl")

#Parallellism
include("parallellism/MultithreadEnv.jl")
include("parallellism/MpiEnv.jl")

end