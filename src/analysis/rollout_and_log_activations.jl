import Dates
import TOML
include("../utils/parse_config.jl")
include("../utils/profiler.jl")
include("../environments/environments.jl")
include("../networks/networks.jl")
include("../algorithms/algorithms.jl")

import PythonCall
import MuJoCo
import Wandb
import BSON
import Flux
import DataStructures
import HDF5
using ProgressMeter
using ComponentArrays: ComponentArray, getdata
include("../utils/wandb_logger.jl")


exploration = false
#wandb_run_id = ARGS[1]
wandb_run_id = "w54we3a2"
base_path = "/home/emil/Development/activation-analysis/local_rollouts"
#base_path = "/n/holylabs/LABS/olveczky_lab/Lab/virtual_rodent/julia_rollout/"
params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")

walker = Environments.Rodent(;merge(params.physics, (;body_scale=1.0))...)
if length(ARGS) >= 2
    animal_session = ARGS[2]
    input_filename  = "$base_path/$animal_session/precomputed_inputs.h5"
    output_filename = "$base_path/$animal_session/$wandb_run_id.h5"
    target = Environments.load_imitation_target(walker, input_filename)
    training_data = false
else
    output_filename = "$base_path/eval_on_training_data/$wandb_run_id.h5"
    target = Environments.load_imitation_target(walker)
    training_data = true
end

falloffs = params.reward.falloff
falloffs = (;com=falloffs.com, rotation=falloffs.rotation, joint=falloffs.joint, joint_vel=falloffs.joint_vel, appendages=falloffs.appendages)
reward_params = merge(params.reward, (;falloff=falloffs))
reward_spec = Environments.EqualRewardWeights(;reward_params...)

env = Environments.ImitationEnv(walker, reward_spec, target; merge(params.imitation, (;target_fps=50.0))...)
if haskey(params, :mod)
    if haskey(params.mod, :simplified_target) && params.mod.simplified_target
        env = Environments.SimplifiedTarget(env)
    end
    if haskey(params.mod, :imitation_speedup_range)
        env = Environments.FPSMod(env, float(params.mod.imitation_speedup_range))
    end
end

params = merge(params, (;network=merge(params.network, (;bottleneck = Symbol(params.network.bottleneck)))))
params = merge(params, (;network=merge(params.network, (;decoder_type = Symbol(params.network.decoder_type)))))

#Hacky code to load the networks even when decoder state is stored
networks = Networks.EncDec(env, params)
decoder_input_size = size(first(networks.decoder).cell.Wi, 2)
training_n_envs = params.rollout.n_envs
Networks.rollout!(networks.decoder, zeros(decoder_input_size, training_n_envs), zeros(Bool, training_n_envs))
network_state = BSON.load(weights_file_name)[:model_state]
Flux.loadmodel!(networks, network_state)
Flux.reset!(networks)
networks_gpu = networks |> Flux.gpu

network_activations = DataStructures.DefaultDict{String, Vector{Matrix{Float32}}}(Vector{Matrix{Float32}})
recordable_networks = Networks.recordable(networks_gpu) do k,v
    push!(network_activations[join(k, "/")], v)
end

_, clip_steps, n_clips = size(env.base_env.target)
env_steps = 2*(clip_steps - params.imitation.horizon)
params = merge(params, (;
    imitation = merge(params.imitation, (;restart_on_reset = false)),
    rollout   = merge(params.rollout,   (;n_steps_per_epoch = env_steps)),
    network   = merge(params.network,   (;bottleneck = Symbol(params.network.bottleneck),
                                          decoder_type = Symbol(params.network.decoder_type))),
))
if !haskey(params.reward, :energy_cost)
    params = merge(params, (; reward=merge(params.reward, (;energy_cost=0.0))))
end
if !exploration
    params = merge(params, (;
        network   = merge(params.network,(;sigma_min = eps(Float32), sigma_max = eps(Float32))),
    ))
end

multienv = Environments.MultithreadEnv(env, n_clips)
for (i, subenv) in enumerate(multienv.environments)
    Environments.reset!(subenv.base_env, i, 1)
end
Flux.reset!(networks_gpu)
collector = Algorithms.CuCollector(multienv, networks, env_steps)

Algorithms.collect_epoch!(collector, networks_gpu, false)
clip_labels = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    [HDF5.attrs(fid["clip_$(i-1)"])["action"] for i=1:n_clips]
end

function rec(fcn, pre, ct)
    axs = getaxes(ct)
    if axs == ()
        fcn(pre, ct)
    else
        for k in keys(axs[1])
            rec(fcn, "$pre/$k", getproperty(ct, k))
        end
    end
end

mkpath(dirname(output_filename))
HDF5.h5open(output_filename, "w") do fid
    write_to_fid(k, v::ComponentArray) =(fid[k] = Flux.cpu(getdata(v)))# (fid[k] = Flux.cpu(getdata(v)) |> Flux.cpu)
    write_to_fid(k, v::AbstractArray) = (fid[k] = Flux.cpu(v) |> Array)
    #write_to_fid(k, v::AbstractArray{T, 3}) where T = (fid[k] = Flux.cpu(v) |> Flux.cpu) 
    #write_to_fid(k, v::AbstractMatrix) = (fid[k] = Flux.cpu(v))
    rec(write_to_fid, "states", collector.states)
    rec(write_to_fid, "infos", collector.infos)
    write_to_fid("status", collector.status)
    write_to_fid("rewards", collector.rewards)
    for (k, v) in pairs(network_activations)
        write_to_fid("activations/$k", cat(v...; dims=3))
    end
    if training_data
       fid["clip_labels"] = clip_labels
    end
end
