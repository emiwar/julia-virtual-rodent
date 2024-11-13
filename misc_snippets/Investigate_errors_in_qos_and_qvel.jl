import Wandb
import Flux
import PythonCall
import BSON
import CUDA
import MuJoCo
import ProgressMeter
include("../src/utils/component_tensor.jl")
include("../src/environments/rodent_imitation_env.jl")
include("../src/algorithms/ppo_networks.jl")
include("../src/utils/wandb_logger.jl")

T = 500
wandb_run_id = "7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

MuJoCo.init_visualiser()

env = RodentImitationEnv(params)
reset!(env, params, next_clip=100)
physics_states = zeros(env.model.nq + env.model.nv + env.model.na,
                       T*params.physics.n_physics_steps)
qpos = zeros(env.model.nq, T)
qvel = zeros(env.model.nv, T)

exploration = false
n_physics_steps = params.physics.n_physics_steps
ProgressMeter.@showprogress for t=1:T
    env_state = state(env, params) |> ComponentTensor
    actor_output = actor(actor_critic, ComponentTensor(CUDA.cu(data(env_state)), index(env_state)), params)
    env.data.ctrl .= clamp.(exploration ? actor_output.action : actor_output.mu, -1.0, 1.0) |> Array
    for tt=1:n_physics_steps
        #dubbleData.qpos[1:(env.model.nq)] .= env.data.qpos
        #dubbleData.qpos[(env.model.nq+1):end] = env.target.qpos[:, target_frame(env), env.target_clip]
        #MuJoCo.forward!(dubbleModel, dubbleData)
        physics_states[:,(t-1) * n_physics_steps + tt] = MuJoCo.get_physics_state(env.model, env.data)
        MuJoCo.step!(env.model, env.data)
    end
    qpos[:, t] .= env.data.qpos
    qvel[:, t] .= env.data.qvel
    env.lifetime += 1
    if t%2==0
        env.target_frame += 1
    end
    if status(env, params) != RUNNING
        reset!(env, params, next_clip=100)
    end
end

new_data = MuJoCo.init_data(env.model)
MuJoCo.visualise!(env.model, new_data, trajectories = physics_states)

import Plots
target = view(env.target, :, :, env.target_clip)

p = Plots.plot(title="Root pos")
for i=1:3
    label = "xyz"[i]
    color = Plots.get_color_palette(:auto, 1)[i]
    Plots.plot!(p, 0.02*(1:250), @view target.qpos[:root_pos, :][i, :]; label, color)
    Plots.plot!(p, 0.01*(1:500), qpos[i, :]; label=label*"_target", color, linestyle=:dash)
end
p
#Plots.show(p)

p = Plots.plot(title="Root vel", xlim=(2.5, 2.7))
for i=1:3
    label = "xyz"[i]
    color = Plots.get_color_palette(:auto, 1)[i]
    Plots.plot!(p, 0.02*(1:250), @view target.qvel[:root_vel, :][i, :]; label, color)
    Plots.plot!(p, 0.01*(1:500), qvel[i, :]; label=label*"_target", color, linestyle=:dash)
end
p
#Plots.show(p)

p = Plots.plot(title="Joints", legend=false, size=(500, 1000), ylim=(125, 130))
for i=8:size(qpos, 1)
    Plots.plot!(p, 0.02*(1:250), array(target.qpos)[i, :] .+ 2i; color=:gray)
    Plots.plot!(p, 0.01*(1:500), qpos[i, :] .+ 2i)
end
p

p = Plots.plot(title="Jointvel", legend=false, size=(500, 1000))
for i=9:size(qvel, 1)
    Plots.plot!(p, 0.02*(1:250), array(target.qvel)[i, :] .+ 20i; color=:gray)
    Plots.plot!(p, 0.01*(1:500), qvel[i, :] .+ 20i)
end
p