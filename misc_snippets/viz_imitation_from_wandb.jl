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

T = 5000
wandb_run_id = "vo055vvi"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

MuJoCo.init_visualiser()

env = RodentImitationEnv(params)
reset!(env, params)
physics_states = zeros(env.model.nq + env.model.nv + env.model.na,
                       T*params.physics.n_physics_steps)

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
    env.lifetime += 1
    if t%2==0
        env.target_frame += 1
    end
    if status(env, params) != RUNNING
        reset!(env, params)
    end
end

new_data = MuJoCo.init_data(env.model)
MuJoCo.visualise!(env.model, new_data, trajectories = physics_states)