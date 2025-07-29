module Networks
import Random

import CUDA
using ComponentArrays: ComponentArray, ComponentVector, getdata, getaxes
using Flux
using ConcreteStructs

import ..Environments

include("actor_critic.jl")

include("enc_dec.jl")
include("info_bottleneck.jl")
include("variational_bottleneck.jl")

include("action_samplers.jl")
include("network_recorder.jl")
include("utils.jl")

include("parallell_control.jl")
include("foot_pos_net.jl")
include("nervenet_like.jl")

end
