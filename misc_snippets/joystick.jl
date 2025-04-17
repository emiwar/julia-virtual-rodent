params = (
    network = (
        actor_size=[1024, 2014],
        critic_size=[1024, 1024],
        sigma_min=2f-1,
        sigma_max=1f0,#5f-1,x
        latent_dimension=60,
        latent_action_scale=2f0
    ),
    physics = (
        n_physics_steps = 5,
        min_torso_z = 0.035,
        spawn_z_offset = 0.01,
        torque_control = true,
        body_scale = 1.0,
        timestep = 0.002,
        foot_mods = true,
        hip_mods = false
    ),
    reward = (
        alive_bonus = 0.1,
        control_cost = 0.000,
        falloff = (
            forward_speed = 0.4,
            turning_speed = 1.0,
            head_height = 0.025,
        ),
	    forward_weight = "auto"
    ),
    training = (
        loss_weight_actor = 1.0,
        loss_weight_critic = 0.5,
        loss_weight_entropy = -0.01,
        n_miniepochs=1,
        learning_rate=1e-4,
        gamma=0.95,
        lambda=0.95,
        clip_range=0.2,
        checkpoint_interval=1000,
    ),
    rollout = (
        n_envs=512,
        n_steps_per_epoch=16,
        n_epochs=100_000,
        reset_on_epoch_start=false,
    )
)

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

#joystick_model = Chain(Dense(3 => 1024, tanh), Dense(1024=>1024, tanh), Dense(1024=>60))
joystick_model = BSON.load("joystick_model3.bson")[:joystick_model] |> Flux.gpu
#params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*", project="emiwar-team/Rodent-Joystick")
#actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

env = RodentJoystickEnv(params)#RodentRunEnv(params)
reset!(env, params)

ActorCritic = VariationalEncDec
pretrained_network, step = load_actor_critic_from_wandb("j0zwbgns")

MuJoCo.init_visualiser()
exploration = false
latent_ou_noise = CUDA.zeros(60)
function control!(model, data)
    env_state = state(env, params) |> ComponentTensor
    joystick_noise, joystick_turn, joystick_backwards, joystick_rear = GLFW.GetJoystickAxes(GLFW.JOYSTICK_1)[3:6]
    joystick_forward = -0.9joystick_backwards#clamp(-0.5joystick_backwards, -.6, 1.0)
    command = Flux.gpu([joystick_forward, -3joystick_turn, 1.2 + 0.6*joystick_rear])
    #println(command |> cpu)
    #command_state = ComponentTensor((command = command))
    #actor_output = joystick_model(command)#actor(actor_critic, command_state, params)
    #motor_command = exploration ? actor_output.action : actor_output.mu
    motor_command = joystick_model(command)
    latent_ou_noise .= 0.98 * latent_ou_noise + 0.05 * CUDA.randn(60) * (1+joystick_noise)
    motor_command .+= latent_ou_noise
    motor_action = decoder_only(pretrained_network, env_state, motor_command, params)
    env.last_torso_pos  = body_xpos(env,  "walker/torso")
    env.last_torso_quat = body_xquat(env, "walker/torso")
    data.ctrl .= clamp.(motor_action, -1.0, 1.0) |> Array
    #MuJoCo.step!(model, data)
    #MuJoCo.step!(model, data)
    #println(getindex(subtree_com(env, "walker/skull"), 3) * 10.0)
    #for tt=1:n_physics_steps
    #    MuJoCo.step!(env.model, env.data)
    #end
    #new_quat = body_xquat(env, "walker/torso")
    #env.lifetime += 1
    buttons = Bool.(GLFW.GetJoystickButtons(GLFW.JOYSTICK_1))
    if status(env, params) != RUNNING || buttons[2]
        reset!(env, params)#, 1, env.target_frame)
    end
end

MuJoCo.visualise!(env.model, env.data; controller=control!)


#while true
#    println(GLFW.GetJoystickAxes(GLFW.JOYSTICK_1))
#    sleep(.1)
#end 