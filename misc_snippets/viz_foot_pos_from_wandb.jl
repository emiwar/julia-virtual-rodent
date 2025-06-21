import Dates
import TOML
include("../src/utils/parse_config.jl")
include("../src/utils/profiler.jl")
include("../src/environments/environments.jl")
include("../src/networks/networks.jl")
include("../src/algorithms/algorithms.jl")

import PythonCall
import MuJoCo
import Wandb
import BSON
import Flux
using ProgressMeter
using ComponentArrays: ComponentArray, getdata
include("../src/utils/wandb_logger.jl")

T = 1000
wandb_run_id = "v4qkc030" #"624ifrxa" # #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*"; project="emiwar-team/Modular-Imitation")

actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

MuJoCo.init_visualiser()

walker = Environments.ModularRodent(;params.physics...)
imparams = merge(params.imitation, (target_fps = float(params.imitation.target_fps),))
env = Environments.FootPosEnv(walker)#; imparams...)
Environments.reset!(env)
Flux.reset!(actor_critic)
physics_states = zeros(walker.model.nq + walker.model.nv + walker.model.na,
                       T*params.physics.n_physics_steps)
exploration = true
n_physics_steps = params.physics.n_physics_steps
ProgressMeter.@showprogress for t=1:T
    env_state = Environments.state(env) |> ComponentArray |> Flux.gpu;
    actor_output = Networks.actor(actor_critic, env_state, false)
    Environments.set_ctrl!(env, (exploration ? actor_output.action : actor_output.mu) |> Flux.cpu)
    for tt=1:n_physics_steps
        Environments.step!(env.walker)
        physics_states[:,(t-1) * n_physics_steps + tt] = MuJoCo.get_physics_state(walker.model, walker.data)
        #env.target_timepoint[] += Environments.dt(env.walker)
    end
    env.lifetime += 1
    if Environments.status(env) != Environments.RUNNING
        println("Resetting at age $(env.lifetime[]), animation step $(t*walker.n_physics_steps)")
        Environments.reset!(env) ##1, env.target_frame)
        Flux.reset!(actor_critic)
    end
end

new_data = MuJoCo.init_data(walker.model)
MuJoCo.visualise!(walker.model, new_data, trajectories = physics_states)

using PrettyPrinting
Environments.state(env) |> pprint
Environments.compute_rewards(env) |> pprint
