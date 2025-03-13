import CUDA
import HDF5
import LinearAlgebra: norm
import Dates
using Flux
using ProgressMeter
using StaticArrays

include("../src/utils/component_tensor.jl")
include("../src/networks/utils.jl")
include("../src/networks/variational_bottleneck.jl")
include("../src/networks/info_bottleneck.jl")
include("../src/networks/action_samplers.jl")
include("../src/networks/enc_dec.jl")

rollout_filename = "/home/emil/Development/activation-analysis/local_rollouts/eval_on_training_data/s66fv425.h5"

rec(d::HDF5.File) = rec(d["/"])
rec(d::HDF5.Group) = map(p->Symbol(p[1])=>rec(p[2]), pairs(d)) |> NamedTuple
rec(d::HDF5.HDF5.Dataset) = d |> Array
states = HDF5.h5open(f->rec(f["states"]), rollout_filename, "r") |> ComponentTensor
states = @view states[:, :, 1:size(states)[3]-1]
actor_outputs = HDF5.h5open(f->f["activations/decoder/layer_3/c"] |> Array, rollout_filename, "r")
mu, sigma = split_halfway(actor_outputs)
reset_mask = HDF5.h5open(f->Array(f["status"]), rollout_filename, "r") .!== UInt8(0)
reset_mask = @view reset_mask[:, 1:end-1]

Flux.gpu(ct::ComponentTensor) = ComponentTensor(Flux.gpu(data(ct)), index(ct))
states_gpu = states |> Flux.gpu
mu_gpu = mu |> Flux.gpu
reset_mask_gpu = reset_mask |> Flux.gpu

function train!(networks, states, mu, reset_mask, opt_state)
    _, N_tot, T_tot = size(states)
    #local l2_loss
    for i=1:32:(N_tot-32)
        Flux.reset!(networks)
        for t=1:16:(T_tot-16)
            S = view(states, :, i:(i+31), t:(t+15))
            R = view(reset_mask, i:(i+31), t:(t+15))
            y = view(mu, :, i:(i+31), t:(t+15))
            g = Flux.gradient(networks) do networks
                l2_loss = Flux.Losses.mse(actor(networks, S, R).mu, y)
            end
            #println(l2_loss)
            Flux.Optimise.update!(opt_state, networks, g[1])
        end
    end
    Flux.reset!(networks)
    Flux.Losses.mse(actor(networks, states, reset_mask).mu, mu)
end

networks = EncDec([size(states.imitation_target)[1], 1024, 1024],
                  [size(states.proprioception)[1] + 60, 1024, 1024],
                  [1024, 1024],
                  size(mu, 1);
                  bottleneck_type = :deterministic,
                  decoder_type = :LSTM,
                  sigma_min = 0, sigma_max = eps(Float32),
                  latent_dimension = 60, gamma = 0.95)
networks_gpu = networks |> Flux.gpu
opt_state = Flux.setup(Flux.Adam(), networks_gpu)
losses = Float32[]
for epoch=1:500
    loss = train!(networks_gpu, states_gpu, mu_gpu, reset_mask_gpu, opt_state)
    push!(losses, loss)
    println("[$(Dates.now())] Epoch $epoch: $loss")
end
