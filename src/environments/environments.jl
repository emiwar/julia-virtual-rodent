module Environments
import MuJoCo
import PythonCall
import HDF5
import LinearAlgebra: norm, dot
import MPI
using StaticArrays
using ..ComponentTensors
using ..Timers: lap

include("abstract_env.jl")

include("../utils/mujoco_quat.jl")

#Imitation
include("imitation/walker.jl")
include("imitation/rodent.jl")
include("imitation/imitation_utils.jl")
include("imitation/imitation_env.jl")
include("imitation/imitation_reward_spec.jl")
include("imitation/fps_mod.jl")

#Parallellism
include("parallellism/multithread_env.jl")
include("parallellism/mpi_env.jl")

end