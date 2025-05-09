import BSON
import CUDA
import Dates
import Statistics
import HDF5
import LinearAlgebra: norm, dot
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
include("../src/collectors/batch_stepper.jl")
include("../src/collectors/mpi_stepper.jl")
include("../src/collectors/cuda_collector.jl")
include("../src/algorithms/ppo.jl")
include("../src/networks/utils.jl")
include("../src/networks/variational_bottleneck.jl")
include("../src/networks/info_bottleneck.jl")
include("../src/networks/action_samplers.jl")
include("../src/networks/enc_dec.jl")

T = 2000
wandb_run_id = "3eciud0m" #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
#ActorCritic = VariationalEncDec
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

MuJoCo.init_visualiser()

target_data = "/home/emil/Development/activation-analysis/local_rollouts/custom_eval1/precomputed_inputs.h5"
env = RodentImitationEnv(params; target_data)#, target_data="reference_data/2020_12_22_1_precomputed.h5")
clip_labels = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    [HDF5.attrs(fid["clip_$(i-1)"])["action"] for i=1:size(env.target)[3]]
end

clips = 1:3#1:842 #findall(l->l=="FastWalk", clip_labels) #FaceGroom #

reset!(env, params, rand(clips), 1) #1, 25000)
Flux.reset!(actor_critic)
dubbleModel = dm_control_model_with_ghost(torque_actuators = params.physics.torque_control,
                                          foot_mods = params.physics.foot_mods,
                                          scale = params.physics.body_scale)
physics_states = zeros(dubbleModel.nq + dubbleModel.nv + dubbleModel.na,
                       T*params.physics.n_physics_steps)
dubbleData = MuJoCo.init_data(dubbleModel)
exploration = false
n_physics_steps = params.physics.n_physics_steps
ProgressMeter.@showprogress for t=1:T
    env_state = state(env, params) |> ComponentTensor
    actor_output = actor(actor_critic, ComponentTensor(CUDA.cu(data(env_state)), index(env_state)), false)
    env.data.ctrl .= clamp.(exploration ? actor_output.action : actor_output.mu, -1.0, 1.0) |> Array
    for tt=1:n_physics_steps
        dubbleData.qpos[1:(env.model.nq)] .= env.data.qpos
        dubbleData.qpos[(env.model.nq+1):end] = array(@view env.target.qpos[:, target_frame(env), env.target_clip])
        MuJoCo.forward!(dubbleModel, dubbleData)
        physics_states[:,(t-1) * n_physics_steps + tt] = MuJoCo.get_physics_state(dubbleModel, dubbleData)
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    if t%2==0
        env.target_frame += 1
    end
    if status(env, params) != RUNNING
        println("Resetting at age $(env.lifetime), frame $(env.target_frame), animation step $(t*n_physics_steps)")
        reset!(env, params, rand(clips), 1) ##1, env.target_frame)
        Flux.reset!(actor_critic)
    end
end

new_data = MuJoCo.init_data(dubbleModel)
MuJoCo.visualise!(dubbleModel, new_data, trajectories = physics_states)
