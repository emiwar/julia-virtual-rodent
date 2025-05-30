module Algorithms
export Collector, CuCollector, ppo

import CUDA
import Flux
import Statistics
using ProgressMeter

using ComponentArrays: ComponentArray, FlatAxis, getaxes, getdata
using ConcreteStructs
using ..Environments
using ..Networks
using ..Timers: lap

include("collectors.jl")
include("ppo.jl")

end