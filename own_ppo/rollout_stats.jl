import BSON
import Flux
import CUDA
import MuJoCo
import Plots
include("mujoco_env.jl")
include("networks.jl")

filename = "runs/checkpoints/test-2024-06-14T15:36:56.427/step-2000.bson"
T = 1000

actor_critic = BSON.load(filename)[:actor_critic] |> Flux.gpu
env = RodentEnv()
reset!(env)
nx = env.model.nq + env.model.nv + env.model.na
physics_states = zeros(nx, T)
ctrl = zeros(env.model.nu, T)
actuator_forces = zeros(env.model.na, T)

for t=1:T
    env_state = map(s->reshape([s;], size(s)..., 1), state(env, params)) |> Flux.gpu
    actor_output = actor(actor_critic, env_state, params)
    act!(env, Flux.cpu(actor_output.action), params)
    ctrl[:, t] .= env.data.ctrl
    actuator_forces[:, t] .= env.data.actuator_force
    if is_terminated(env, params)
        reset!(env)
    end
end

function plot_traces(traces; labels=nothing, kwargs...)
    p = Plots.plot(legend=false, size=(400, 1000),
                   axis=false; kwargs...)
    for i=1:size(traces, 1)
        Plots.plot!(size(traces, 2).*[0.0, 1.02],
                    [3(i-1), 3(i-1)], color=:black, linestyle=:dot)
        Plots.plot!(p, traces[i, :] .+ 3(i-1))
        Plots.annotate!(p, 0, 3(i-1),
                        Plots.text(actuator_name(i), 8, :right,
                                   color=p.series_list[end][:linecolor]))
    end
    Plots.plot!(p, [75, 100], [-1.5, -1.5], color=:black)
    Plots.annotate!(p, 87.5, -2.5, ("25ms", 8))
    return p
end

plot_traces(ctrl, xlim=(-35, 100), title="CTRL")
plot_traces(actuator_forces, xlim=(-35, 100), title="actuator_forces")

function actuator_name(id)
    char_ptr = MuJoCo.mj_id2name(env.model, MuJoCo.mjOBJ_ACTUATOR, id-1)
    if char_ptr == C_NULL
        return "<Invalid>"
    else
        return unsafe_string(char_ptr)
    end
end