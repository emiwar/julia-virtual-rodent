module Environments
import MuJoCo
import PythonCall
import HDF5
import LinearAlgebra: norm, dot
import MPI
using StaticArrays
using ConcreteStructs
import ComponentArrays
using ComponentArrays: ComponentArray, ComponentVector, getdata, getaxes, FlatAxis
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

include("modular_imitation/modular_rodent.jl")
include("modular_imitation/modular_imitation.jl")
include("modular_imitation/modular_standup.jl")
include("modular_imitation/foot_pos_env.jl")

#Joystick
include("joystick_env.jl")
include("move_to_target_env.jl")

#Parallellism
include("parallellism/multithread_env.jl")
include("parallellism/mpi_env.jl")

end