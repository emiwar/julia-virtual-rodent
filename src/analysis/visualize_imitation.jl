import BSON
import Flux
import CUDA
import MuJoCo
import HDF5
import ProgressMeter
include("../utils/component_tensor.jl")
include("../environments/rodent_imitation_env.jl")
include("../algorithms/ppo_networks.jl")
MuJoCo.init_visualiser()


dubbleModelPath = "src/environments/assets/imitation_viz_scale080.xml"
dubbleModel = MuJoCo.load_model(dubbleModelPath)
dubbleData = MuJoCo.init_data(dubbleModel)
#qpos_targets = HDF5.h5open(fid->(fid["qpos"][:, :]), "src/environments/assets/com_trajectory2.h5", "r")


runname = "RodentComAndDirImitation-2024-08-23T18:36:56.706"
params = HDF5.h5open(fid->NamedTuple(Symbol(k)=>v[] for (k,v) in pairs(fid["params"])), "runs/$runname.h5", "r")
filename = "runs/checkpoints/$runname/step-20000.bson"
T = 10000

actor_critic = BSON.load(filename)[:actor_critic] |> Flux.gpu

env = RodentImitationEnv()
reset!(env, params)
nx = dubbleModel.nq + dubbleModel.nv + dubbleModel.na
physics_states = zeros(nx, T*params.n_physics_steps)
com = zeros(3, T)
com_target = zeros(3, T)
dist_to_target = zeros(T)
ang_to_target = zeros(T)
actions = zeros(env.model.nu, T)
ProgressMeter.@showprogress for t=1:T
    env_state = state(env, params)
    actor_output = actor(actor_critic, ComponentTensor(CUDA.cu(data(env_state)), index(env_state)), params)
    actions[:, t] .= clamp.(actor_output.mu, -1.0, 1.0) |> Array#clamp.(actor_output.action, -1.0, 1.0) |> Array
    env.data.ctrl .= actions[:, t]
    for tt=1:params.n_physics_steps
        dubbleData.qpos[1:(env.model.nq)] .= env.data.qpos
        dubbleData.qpos[(env.model.nq+1):end] = env.target.qpos[:, target_frame(env)]
        MuJoCo.forward!(dubbleModel, dubbleData)
        physics_states[:,(t-1)*params.n_physics_steps + tt] = MuJoCo.get_physics_state(dubbleModel, dubbleData)
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    if t%2==0
        env.target_frame += 1
    end
    com[:, t] = MuJoCo.body(env.data, "torso").com
    com_target[:, t] = env.target.com[:, target_frame(env)]
    envinfo = info(env)
    dist_to_target[t] = envinfo.target_distance[1]
    ang_to_target[t] = envinfo.angle_to_target[1]
    if is_terminated(env, params)
        reset!(env, params)
    end
end

new_data = MuJoCo.init_data(dubbleModel)
MuJoCo.visualise!(dubbleModel, new_data, trajectories = physics_states)

import Plots

Plots.plot(angle_to_target)
Plots.plot(com_target[1, :])

