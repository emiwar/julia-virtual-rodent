import Dates
import TOML
include("../src/utils/parse_config.jl")
include("../src/utils/profiler.jl")
include("../src/environments/environments.jl")
include("../src/networks/networks.jl")
include("../src/algorithms/algorithms.jl")

import PythonCall
import MuJoCo
import Wandb
import BSON
import Flux
using ProgressMeter
using ComponentArrays: ComponentArray, getdata
include("../src/utils/wandb_logger.jl")

T = 2000
wandb_run_id = "yk9qnl1p" #"624ifrxa" # #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*"; project="emiwar-team/Modular-Imitation")

actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

MuJoCo.init_visualiser()

#target_data = "/home/emil/Development/activation-analysis/local_rollouts/custom_eval1/precomputed_inputs.h5"
#env = RodentImitationEnv(params; target_data)#, target_data="reference_data/2020_12_22_1_precomputed.h5")
#clip_labels = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
#    [HDF5.attrs(fid["clip_$(i-1)"])["action"] for i=1:size(env.target)[3]]
#end

#Setup the environment
env_params = merge(params.imitation, (;target_fps=float(params.imitation.target_fps)))
walker = Environments.ModularRodent(;params.physics...)
env = Environments.ModularImitationEnv(walker; env_params...)

Environments.reset!(env)#, rand(clips), 1) #1, 25000)
Flux.reset!(actor_critic)
dubbleModel = Environments.dm_control_rodent_with_ghost()#torque_actuators = params.physics.torque_control,
                                          #foot_mods = params.physics.foot_mods,
                                          #scale = params.physics.body_scale)
physics_states = zeros(dubbleModel.nq + dubbleModel.nv + dubbleModel.na,
                       T*params.physics.n_physics_steps)
dubbleData = MuJoCo.init_data(dubbleModel)
exploration = false
n_physics_steps = params.physics.n_physics_steps
ProgressMeter.@showprogress for t=1:T
    env_state = Environments.state(env) |> ComponentArray |> Flux.gpu
    actor_output = Networks.actor(actor_critic, env_state, false)
    Environments.set_ctrl!(env.walker, Flux.cpu(actor_output.action))
    #base_env = env.base_env
    #walker = base_env.walker
    #walker.data.ctrl .= clamp.(exploration ? actor_output.action : actor_output.mu, -1.0, 1.0) |> Array
    for tt=1:n_physics_steps
        dubbleData.qpos[1:(walker.model.nq)] .= walker.data.qpos
        dubbleData.qpos[(walker.model.nq+1):end] = @view env.target_poses.qpos[:, Environments.target_frame(env), Environments.target_clip(env)]
        MuJoCo.forward!(dubbleModel, dubbleData)
        physics_states[:,(t-1) * n_physics_steps + tt] = MuJoCo.get_physics_state(dubbleModel, dubbleData)
        MuJoCo.step!(walker.model, walker.data)
    end
    env.target_timepoint[] += Environments.dt(walker) * walker.n_physics_steps
    env.lifetime[] += 1
    if Environments.status(env) != Environments.RUNNING
        println("Resetting at age $(env.lifetime[]), frame $(Environments.target_frame(env)), animation step $(t*walker.n_physics_steps)")
        Environments.reset!(env) ##1, env.target_frame)
        Flux.reset!(actor_critic)
    end
end

new_data = MuJoCo.init_data(dubbleModel)
MuJoCo.visualise!(dubbleModel, new_data, trajectories = physics_states)
