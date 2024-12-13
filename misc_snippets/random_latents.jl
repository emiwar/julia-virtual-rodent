import Wandb
import Flux
import PythonCall
import BSON
import CUDA
import MuJoCo
import ProgressMeter
include("../src/utils/component_tensor.jl")
include("../src/environments/imitation_trajectory.jl")
include("../src/environments/rodent_imitation_env.jl")
include("../src/networks/variational_enc_dec.jl")
include("../src/utils/wandb_logger.jl")
include("../src/utils/load_dm_control_model.jl")

T = 2000
wandb_run_id = "j0zwbgns" #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

MuJoCo.init_visualiser()

env = RodentImitationEnv(params)
reset!(env, params)#; next_clip=449)
model = env.model
physics_states = zeros(model.nq + model.nv + model.na,
                       T*params.physics.n_physics_steps)
exploration = false
n_physics_steps = params.physics.n_physics_steps
latent = CUDA.zeros(params.network.latent_dimension)
ProgressMeter.@showprogress for t=1:T
    env_state = state(env, params) |> ComponentTensor
    latent = 0.9f0*latent + 0.2f0 .* CUDA.randn(params.network.latent_dimension)
    env_state_cu = ComponentTensor(CUDA.cu(data(env_state)), index(env_state))
    action = decoder_only(actor_critic, env_state_cu, latent, params)
    env.data.ctrl .= clamp.(action, -1.0, 1.0) |> Array
    for tt=1:n_physics_steps
        physics_states[:,(t-1) * n_physics_steps + tt] = MuJoCo.get_physics_state(model, env.data)
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
end

new_data = MuJoCo.init_data(model)
MuJoCo.visualise!(model, new_data, trajectories = physics_states)