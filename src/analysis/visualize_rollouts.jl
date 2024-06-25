import BSON
import Flux
import CUDA
import MuJoCo
include("../environments/rodent_imitation_env.jl")
include("../algorithms/ppo_networks.jl")
MuJoCo.init_visualiser()

runname = "test-2024-06-24T20:03:18.847"
params = HDF5.h5open(fid->NamedTuple(Symbol(k)=>v[] for (k,v) in pairs(fid["params"])), "runs/$runname.h5", "r")
filename = "runs/checkpoints/$runname/step-3000.bson"
T = 1000

actor_critic = BSON.load(filename)[:actor_critic] |> Flux.gpu

env = RodentImitationEnv()
reset!(env)
nx = env.model.nq + env.model.nv + env.model.na
physics_states = zeros(nx, T*params.n_physics_steps)
com = zeros(3, T)
dist_to_target = zeros(T)
for t=1:T
    env_state = map(s->reshape([s;], size(s)..., 1), state(env, params)) |> CUDA.cu
    actor_output = actor(actor_critic, env_state, params)
    env.data.ctrl .= clamp.(actor_output.action, -1.0, 1.0) |> Array
    for tt=1:params.n_physics_steps
        physics_states[:,(t-1)*params.n_physics_steps + tt] = MuJoCo.get_physics_state(env.model, env.data)
        MuJoCo.step!(env.model, env.data)
    end
    com[:, t] = MuJoCo.body(env.data, "torso").com
    target_vec = get_target_vector(env, params)
    dist_to_target[t] = LinearAlgebra.norm(target_vec)
    env.lifetime += 1
    if is_terminated(env, params)
        reset!(env)
    end
end

new_data = MuJoCo.init_data(env.model)
MuJoCo.visualise!(env.model, new_data, trajectories = physics_states)

p = Plots.plot()
for i in 1:3
    Plots.plot!(p, physics_states[i, :])
    Plots.plot!(p, 5 .* (1:1000), com[i, :])
end
p