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
qpos_targets = HDF5.h5open(fid->(fid["qpos"][:, :]), "src/environments/assets/com_trajectory2.h5", "r")


runname = "RodentComAndDirImitation-2024-07-03T11:16:48.443"
params = HDF5.h5open(fid->NamedTuple(Symbol(k)=>v[] for (k,v) in pairs(fid["params"])), "runs/$runname.h5", "r")
filename = "runs/checkpoints/$runname/step-11000.bson"
T = 1000

actor_critic = BSON.load(filename)[:actor_critic] |> Flux.gpu

env = RodentImitationEnv()
reset!(env)
nx = dubbleModel.nq + dubbleModel.nv + dubbleModel.na
physics_states = zeros(nx, T*params.n_physics_steps)
com = zeros(3, T)
com_target = zeros(3, T)
dist_to_target = zeros(T)
angle_to_target = zeros(T)
ProgressMeter.@showprogress for t=1:T
    env_state = state(env, params)
    actor_output = actor(actor_critic, ComponentTensor(CUDA.cu(data(env_state)), index(env_state)), params)
    env.data.ctrl .= clamp.(actor_output.action, -1.0, 1.0) |> Array
    for tt=1:params.n_physics_steps
        dubbleData.qpos[1:(env.model.nq)] .= env.data.qpos
        dubbleData.qpos[(env.model.nq+1):end] = qpos_targets[:, env.lifetime ÷ 2 + 1]
        MuJoCo.forward!(dubbleModel, dubbleData)
        physics_states[:,(t-1)*params.n_physics_steps + tt] = MuJoCo.get_physics_state(dubbleModel, dubbleData)
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    com[:, t] = MuJoCo.body(env.data, "torso").com
    com_target_ind = env.lifetime ÷ 2 + 1
    com_target[:, t] = env.com_targets[:, com_target_ind]
    envinfo = info(env)
    dist_to_target[t] = envinfo.target_distance[1]
    angle_to_target[t] = envinfo.angle_to_target[1]
    if is_terminated(env, params)
        reset!(env)
    end
end

new_data = MuJoCo.init_data(dubbleModel)
MuJoCo.visualise!(dubbleModel, new_data, trajectories = physics_states)

import Plots

Plots.plot(angle_to_target)