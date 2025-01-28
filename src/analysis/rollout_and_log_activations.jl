import BSON
import CUDA
import Dates
import Statistics
import HDF5
import LinearAlgebra: norm
import MuJoCo
import PythonCall
import Wandb
using Flux
using ProgressMeter
using StaticArrays

include("../utils/profiler.jl")
include("../utils/load_dm_control_model.jl")
include("../utils/mujoco_quat.jl")
include("../utils/component_tensor.jl")
include("../utils/wandb_logger.jl")
include("../environments/mujoco_env.jl")
include("../environments/imitation_trajectory.jl")
include("../environments/rodent_imitation_env.jl")
include("../networks/enc_dec.jl")
include("../collectors/batch_stepper.jl")
include("../collectors/mpi_stepper.jl")
include("../collectors/cuda_collector.jl")

exploration = false
wandb_run_id = ARGS[1] #"mm0dnyq4"
output_filename = "/n/holylabs/LABS/olveczky_lab/Lab/virtual_rodent/julia_rollout/eval_on_training_data/$wandb_run_id.h5"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
VariationalEncDec = EncDec
networks = BSON.load(weights_file_name)[:actor_critic]
networks_gpu = networks |> Flux.gpu

template_env = RodentImitationEnv(params)
_, clip_steps, n_clips = size(template_env.target)
env_steps = 2*(clip_steps - params.imitation.horizon)
params = merge(params, (;
    imitation = merge(params.imitation, (;restart_on_reset = false)),
    rollout   = merge(params.rollout,   (;n_steps_per_epoch = env_steps)),
))

stepper = BatchStepper(template_env, n_clips)
for (i, env) in enumerate(stepper.environments)
    reset!(env, params, i, 1)
end   
collector = CuCollector(template_env, networks, n_clips, env_steps; actor_fcn=actor_logged)
lapTimer = LapTimer()
collect_batch!(collector, stepper, params, lapTimer) do state, params
    actor_logged(networks_gpu, state, params)
end

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
    write_to_fid(k, v::ComponentTensor) = (fid[k] = permutedims(array(v) |> Array, (2,3,1)))
    write_to_fid(k, v::AbstractArray{T, 3}) where T = (fid[k] = permutedims(v |> Array, (2,3,1)))
    write_to_fid(k, v::AbstractMatrix) = (fid[k] = permutedims(v |> Array, (2,1)))
    rec(write_to_fid, "states", collector.states)
    rec(write_to_fid, "infos", collector.infos)
    write_to_fid("status", collector.status)
    write_to_fid("rewards", collector.rewards)
    for i in 1:length(networks.encoder)
        write_to_fid("activations/encoder/layer_$i", view(collector.actor_outputs, Symbol("encoder_", i)))
    end
    for i in 1:length(networks.decoder)
        write_to_fid("activations/decoder/layer_$i", view(collector.actor_outputs, Symbol("decoder_", i)))
    end
    write_to_fid("activations/latent_layer", collector.actor_outputs.latent)
    write_to_fid("activations/output/mean", collector.actor_outputs.mu)
    write_to_fid("activations/output/std", collector.actor_outputs.sigma)
    fid["clip_labels"] = clip_labels
end
