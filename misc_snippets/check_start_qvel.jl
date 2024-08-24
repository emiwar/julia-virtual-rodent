import BSON
import Flux
import CUDA
import MuJoCo
import HDF5
import ProgressMeter
include("../src/utils/component_tensor.jl")
include("../src/environments/rodent_imitation_env.jl")
include("../src/algorithms/ppo_networks.jl")
MuJoCo.init_visualiser()


dubbleModelPath = "src/environments/assets/imitation_viz_scale080.xml"
dubbleModel = MuJoCo.load_model(dubbleModelPath)
dubbleData = MuJoCo.init_data(dubbleModel)

T = 1000

env = RodentImitationEnv()
reset!(env, params)
nx = dubbleModel.nq + dubbleModel.nv + dubbleModel.na
physics_states = zeros(nx, T*params.n_physics_steps)
ProgressMeter.@showprogress for t=1:T
    env_state = state(env, params)
    env.data.ctrl .= 0.0
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
    if env.lifetime > 100
        env.lifetime = 0
        env.cumulative_reward = 0.0
        env.target_frame = rand(1:(length(env.target)-200))
        MuJoCo.reset!(env.model, env.data)
        env.data.qpos .= view(env.target.qpos, :, target_frame(env))
        env.data.qpos[3] += .01
        env.data.qvel .= view(env.target.qvel, :, target_frame(env))
        MuJoCo.forward!(env.model, env.data)
    end
end

new_data = MuJoCo.init_data(dubbleModel)
MuJoCo.visualise!(dubbleModel, new_data, trajectories = physics_states)

import Plots

Plots.plot(angle_to_target)
Plots.plot(com_target[1, :])

