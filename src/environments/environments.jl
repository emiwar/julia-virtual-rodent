module Environments
import MuJoCo
import PythonCall
import HDF5
import LinearAlgebra: norm, dot
import MPI
using StaticArrays
using ComponentArrays: ComponentArray, getdata, getaxes, FlatAxis
using ..Timers: lap

include("abstract_env.jl")

include("../utils/mujoco_quat.jl")

#Imitation
include("imitation/walker.jl")
include("imitation/rodent.jl")
include("imitation/imitation_utils.jl")
include("imitation/imitation_env.jl")
include("imitation/imitation_reward_spec.jl")
include("imitation/mods/fps_mod.jl")
include("imitation/mods/simplified_target.jl")

#Joystick
include("joystick_env.jl")
include("move_to_target_env.jl")

#Parallellism
include("parallellism/multithread_env.jl")
include("parallellism/mpi_env.jl")

end