import BSON
import CUDA
import Dates
import Statistics
import HDF5
import LinearAlgebra: norm, dot
import MuJoCo
import PythonCall
import Wandb
using Flux
using ProgressMeter
using StaticArrays

include("../src/utils/profiler.jl")
include("../src/utils/load_dm_control_model.jl")
include("../src/utils/mujoco_quat.jl")
include("../src/utils/component_tensor.jl")
include("../src/utils/wandb_logger.jl")
include("../src/environments/mujoco_env.jl")
include("../src/environments/rodent_run_env.jl")
include("../src/environments/rodent_joystick_env.jl")
include("../src/collectors/batch_stepper.jl")
include("../src/collectors/mpi_stepper.jl")
include("../src/collectors/cuda_collector.jl")
include("../src/algorithms/ppo_loss.jl")
include("../src/networks/variational_enc_dec.jl")
include("../src/networks/state_invariant.jl")
include("../src/networks/joystick_mlp.jl")

T = 5000
wandb_run_id = "xxcrza2h"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*", project="emiwar-team/Rodent-Joystick")
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu
env = RodentJoystickEnv(params)#RodentRunEnv(params)
reset!(env, params)
last_quat = body_xquat(env, "walker/torso")
MuJoCo.init_visualiser()

ActorCritic = VariationalEncDec
pretrained_network, step = load_actor_critic_from_wandb("j0zwbgns")

physics_states = zeros(env.model.nq + env.model.nv + env.model.na,
                       T*params.physics.n_physics_steps)
exploration = false
n_physics_steps = params.physics.n_physics_steps
fspeed = zeros(T)
tspeed = zeros(T)
ProgressMeter.@showprogress for t=1:T
    env_state = state(env, params) |> ComponentTensor
    actor_output = actor(actor_critic, ComponentTensor(CUDA.cu(data(env_state)), index(env_state)), params)
    motor_command = exploration ? actor_output.action : actor_output.mu
    action = decoder_only(pretrained_network, env_state, motor_command, params)
    env.last_torso_pos  = body_xpos(env,  "walker/torso")
    env.last_torso_quat = body_xquat(env, "walker/torso")
    env.data.ctrl .= clamp.(action, -1.0, 1.0) |> Array
    for tt=1:n_physics_steps
        MuJoCo.step!(env.model, env.data)
        physics_states[:,(t-1) * n_physics_steps + tt] = MuJoCo.get_physics_state(env.model, env.data)
    end
    new_quat = body_xquat(env, "walker/torso")
    fspeed[t] = forward_speed(env, params)
    tspeed[t] = turning_speed(env, params)
    last_quat = new_quat
    env.lifetime += 1
    if status(env, params) != RUNNING
        reset!(env, params)#, 1, env.target_frame)
    end
end

new_data = MuJoCo.init_data(env.model)
MuJoCo.visualise!(env.model, new_data, trajectories = physics_states)

import Plots
Plots.plot(fspeed)
Plots.plot(tspeed)

begin
    Plots.plot(physics_states[1, :], label="Root x")
    Plots.plot!(physics_states[2, :], label="Root y")
    Plots.plot!(physics_states[3, :], label="Root z")
end
nq = env.model.nq
Plots.plot(physics_states[nq+1, :], label="Root vx")
Plots.plot!(physics_states[nq+2, :], label="Root vy")
Plots.plot!(physics_states[nq+3, :], label="Root vz")

Plots.plot(physics_states[nq+4, :], label="Root rot vx")
Plots.plot!(physics_states[nq+5, :], label="Root rot vy")
Plots.plot!(physics_states[nq+6, :], label="Root rot vz")

Plots.plot(physics_states[4, :],  label="Root rot w")
Plots.plot!(physics_states[5, :], label="Root rot x")
Plots.plot!(physics_states[6, :], label="Root rot y")
Plots.plot!(physics_states[7, :], label="Root rot z")

qw = physics_states[4, :]
qx = physics_states[5, :]
qy = physics_states[6, :]
qz = physics_states[7, :]
Plots.plot(atan.(qz, qw)*2)



Plots.plot(rot_speed, ylim=(-10, 10), xlim=(100, 200))