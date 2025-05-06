module Networks
import Random

import CUDA
using ComponentArrays: ComponentArray
using Flux

import ..Environments
#using ..ComponentTensors

include("enc_dec.jl")
include("info_bottleneck.jl")
include("variational_bottleneck.jl")

include("action_samplers.jl")
include("network_recorder.jl")
include("utils.jl")

end
