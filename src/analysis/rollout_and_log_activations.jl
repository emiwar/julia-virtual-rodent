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
import DataStructures
using Flux
using ProgressMeter
using StaticArrays

include("../utils/profiler.jl")
include("../utils/load_dm_control_model.jl")
include("../utils/mujoco_quat.jl")
include("../utils/component_tensor.jl")
include("../utils/wandb_logger.jl")
include("../utils/parse_config.jl")
include("../environments/mujoco_env.jl")
include("../environments/imitation_trajectory.jl")
include("../environments/rodent_imitation_env.jl")
include("../collectors/batch_stepper.jl")
include("../collectors/mpi_stepper.jl")
include("../collectors/cuda_collector.jl")
include("../algorithms/ppo.jl")
include("../networks/utils.jl")
include("../networks/variational_bottleneck.jl")
include("../networks/info_bottleneck.jl")
include("../networks/action_samplers.jl")
include("../networks/enc_dec.jl")
include("../networks/network_recorder.jl")

exploration = false
wandb_run_id = ARGS[1]
base_path = "/n/holylabs/LABS/olveczky_lab/Lab/virtual_rodent/julia_rollout/"
params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")

if length(ARGS) >= 2
    animal_session = ARGS[2]
    input_filename  = "$base_path/$animal_session/precomputed_inputs.h5"
    output_filename = "$base_path/$animal_session/$wandb_run_id.h5"
    template_env = RodentImitationEnv(params, target_data=input_filename)
    training_data = false
else
    output_filename = "$base_path/eval_on_training_data/$wandb_run_id.h5"
    template_env = RodentImitationEnv(params)
    training_data = true
end

networks = BSON.load(weights_file_name)[:actor_critic]
Flux.reset!(networks)
networks_gpu = networks |> Flux.gpu
network_activations = DataStructures.DefaultDict{String, Vector{Matrix{Float32}}}(Vector{Matrix{Float32}})

recordable_networks = recordable(networks_gpu) do k,v
    push!(network_activations[join(k, "/")], v)
end

_, clip_steps, n_clips = size(template_env.target)
env_steps = 2*(clip_steps - params.imitation.horizon)
params = merge(params, (;
    imitation = merge(params.imitation, (;restart_on_reset = false)),
    rollout   = merge(params.rollout,   (;n_steps_per_epoch = env_steps)),
    network   = merge(params.network,   (;bottleneck = Symbol(params.network.bottleneck),
                                          decoder_type = Symbol(params.network.decoder_type))),
))

stepper = BatchStepper(template_env, n_clips)
for (i, env) in enumerate(stepper.environments)
    reset!(env, params, i, 1)
end
collector = CuCollector(template_env, networks, n_clips, env_steps)
Flux.reset!(networks)
lapTimer = LapTimer()
collect_batch!(collector, recordable_networks, stepper, params, lapTimer)
clip_labels = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    [HDF5.attrs(fid["clip_$(i-1)"])["action"] for i=1:n_clips]
end

function rec(fcn, pre, ct)
    if length(keys(ct)) == 1
        fcn(pre, ct)
    else
        for k in keys(ct)
            rec(fcn, "$pre/$k", view(ct, k))
        end
    end
end

mkpath(dirname(output_filename))
HDF5.h5open(output_filename, "w") do fid
    write_to_fid(k, v::ComponentTensor) = (fid[k] = Flux.cpu(array(v)))
    write_to_fid(k, v::AbstractArray{T, 3}) where T = (fid[k] = Flux.cpu(v))
    write_to_fid(k, v::AbstractMatrix) = (fid[k] = Flux.cpu(v))
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
