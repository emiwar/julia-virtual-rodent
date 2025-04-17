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

include("../src/utils/profiler.jl")
include("../src/utils/load_dm_control_model.jl")
include("../src/utils/mujoco_quat.jl")
include("../src/utils/component_tensor.jl")
include("../src/utils/wandb_logger.jl")
include("../src/environments/mujoco_env.jl")
include("../src/environments/imitation_trajectory.jl")
include("../src/environments/rodent_imitation_env.jl")
#include("../src/networks/variational_enc_dec.jl")
include("../src/networks/enc_dec.jl")
include("../src/collectors/batch_stepper.jl")

exploration = false
wandb_run_id = "38h4nifl" #"j0zwbgns" #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
ActorCritic = EncDec#VariationalEncDec
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

template_env = RodentImitationEnv(params)#, target_data="reference_data/2020_12_22_1_precomputed.h5")
stepper = BatchStepper(template_env, size(template_env.target)[3])
for (i, env) in enumerate(stepper.environments)
    reset!(env, params, i, 1)
end
batch_dims = (500, size(template_env.target)[3])
statuses = fill(-1, batch_dims)
infos = BatchComponentTensor((@view stepper.infos[:, 1]), batch_dims...)
prepareEpoch!(stepper, params)
cu(ct::ComponentTensor) = ComponentTensor(CUDA.cu(data(ct)), index(ct))
ProgressMeter.@showprogress for t=1:batch_dims[1]
    actor_output = actor(actor_critic, cu(stepper.states), params)
    copyto!(stepper.actions, exploration ? actor_output.action : actor_output.mu)
    step!(stepper, params)
    infos[:, t, :] = stepper.infos
    statuses[t, :] = stepper.status
end

clip_labels = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    [HDF5.attrs(fid["clip_$(i-1)"])["action"] for i=1:batch_dims[2]]
end

HDF5.h5open("trainset_eval_$(wandb_run_id).h5", "w") do fid
    fid["clip_labels"] = clip_labels
    fid["status"] = statuses
    for k in keys(infos)
        fid["info/$k"] = copy(array(view(infos, k)))
    end
end
