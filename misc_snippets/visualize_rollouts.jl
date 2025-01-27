import BSON
import Flux
import CUDA
import MuJoCo
import HDF5
include("../utils/component_tensor.jl")
include("../environments/rodent_imitation_env.jl")
include("../algorithms/ppo_networks.jl")
MuJoCo.init_visualiser()

runname = "RodentComAndDirImitation-2024-07-02T03:30:48.437"
params = HDF5.h5open(fid->NamedTuple(Symbol(k)=>v[] for (k,v) in pairs(fid["params"])), "runs/$runname.h5", "r")
filename = "runs/checkpoints/$runname/step-36000.bson"
T = 5000

actor_critic = BSON.load(filename)[:actor_critic] |> Flux.gpu

env = RodentImitationEnv()
reset!(env)
nx = env.model.nq + env.model.nv + env.model.na
physics_states = zeros(nx, T*params.n_physics_steps)
com = zeros(3, T)
com_target = zeros(3, T)
dist_to_target = zeros(T)
for t=1:T
    env_state = state(env, params)
    actor_output = actor(actor_critic, ComponentTensor(CUDA.cu(data(env_state)), index(env_state)), params)
    env.data.ctrl .= clamp.(actor_output.action, -1.0, 1.0) |> Array
    for tt=1:params.n_physics_steps
        physics_states[:,(t-1)*params.n_physics_steps + tt] = MuJoCo.get_physics_state(env.model, env.data)
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    com[:, t] = MuJoCo.body(env.data, "torso").com
    com_target_ind = env.lifetime ÷ 2 + 1
    com_target[:, t] = env.com_targets[:, com_target_ind]
    dist_to_target[t] = LinearAlgebra.norm(get_target_vector(env, params))
    if is_terminated(env, params)
        reset!(env)
    end
end

new_data = MuJoCo.init_data(env.model)
MuJoCo.visualise!(env.model, new_data, trajectories = physics_states)

p = Plots.plot()
for i in 1:3
    Plots.plot!(p, physics_states[i, :])
    Plots.plot!(p, 5 .* (1:5000), com[i, :])
end
p

p = Plots.plot()
for i in 1:3
    coord = "xyz"[i]
    color = Plots.color(["red", "blue", "green"][i])
    Plots.plot!(p, com_target[i, 1:5000], label="Target $coord", color=color, linestyle=:dash)
    Plots.plot!(p, com[i, :], label="Output $coord", color=color)
end
p
Plots.plot(dist_to_target, label="Dist to target")
Plots.plot(com[3, :], label="z")

anim = @Plots.animate for t=1:5000
    phi = LinRange(0, 2pi, 1000)
    p = Plots.plot(sin.(2.0 .* phi), cos.(phi) .- 1.0,
                    color=:black, linestyle=:dash,label=false,
                    xlim=(-1.5, 1.5), ylim=(-2.5, 0.5),
                    xticks=[-1, 0, 1], yticks=[0, -1, -2])
    Plots.plot!(p, com[1, 1:t], com[2, 1:t], ratio=1, label=false,
                linewidth=2)
    Plots.scatter!(p, com[1, t:t], com[2, t:t], label=false)
end
Plots.gif(anim, "trace.mp4")