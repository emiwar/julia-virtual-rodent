import BSON
import Flux
import CUDA
import MuJoCo
include("mujoco_env.jl")
include("networks.jl")
MuJoCo.init_visualiser()

params = (;n_physics_steps=5,
           sigma_min=1f-2,
           sigma_max=1f0,
           min_torso_z = 0.035)
filename = "runs/checkpoints/test-2024-06-15T15:03:36.958/step-50000.bson"
T = 1000

actor_critic = BSON.load(filename)[:actor_critic] |> Flux.gpu
env = RodentEnv()
reset!(env)
nx = env.model.nq + env.model.nv + env.model.na
physics_states = zeros(nx, T*params.n_physics_steps)

for t=1:T
    env_state = map(s->reshape([s;], size(s)..., 1), state(env, params)) |> Flux.gpu
    actor_output = actor(actor_critic, env_state, params)
    env.data.ctrl .= clamp.(actor_output.action.ctrl, -1.0, 1.0) |> Flux.cpu
    for tt=1:params.n_physics_steps
        physics_states[:,(t-1)*params.n_physics_steps + tt] = MuJoCo.get_physics_state(env.model, env.data)
        MuJoCo.step!(env.model, env.data)
    end
    if is_terminated(env, params)
        reset!(env)
    end
end

new_data = MuJoCo.init_data(env.model)
MuJoCo.visualise!(env.model, new_data, trajectories = physics_states)