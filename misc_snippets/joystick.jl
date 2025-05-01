import Wandb
import BSON
import Dates
import TOML
import PythonCall
import CUDA
import MuJoCo
using StaticArrays: SVector, SMatrix
using ProgressMeter
include("../src/utils/parse_config.jl")
include("../src/utils/component_tensor.jl")
include("../src/utils/profiler.jl")
include("../src/utils/wandb_logger.jl")
include("../src/environments/environments.jl")
include("../src/networks/networks.jl")
include("../src/networks/utils.jl")
include("../src/algorithms/algorithms.jl")

using .Networks: EncDec, VariationalBottleneck, GaussianActionSampler#, split_halfway
using .ComponentTensors: array

using Flux: Chain, gpu, Dense, Adam
using LinearAlgebra: norm, dot
wandb_run_id = "zf8zs3kq" #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu
joystick_model = BSON.load("joystick_model_linear.bson")[:joystick_model] |> Flux.gpu

#Setup the environment
walker = Environments.Rodent(;merge(params.physics, (;body_scale=1.0))...)
Environments.reset!(walker)
Flux.reset!(actor_critic)

MuJoCo.init_visualiser()
exploration = false
latent_ou_noise = CUDA.zeros(60)
function control!(model, data)
    prop = Environments.proprioception(walker) |> ComponentTensors.ComponentTensor |> array |> gpu
    joystick_noise, joystick_turn, joystick_backwards, joystick_rear = GLFW.GetJoystickAxes(GLFW.JOYSTICK_1)[3:6]
    joystick_forward = -0.9joystick_backwards#clamp(-0.5joystick_backwards, -.6, 1.0)
    command = Flux.gpu([joystick_forward, -3joystick_turn, 1.2 + 0.6*joystick_rear])
    #println(command |> cpu)
    #command_state = ComponentTensor((command = command))
    #actor_output = joystick_model(command)#actor(actor_critic, command_state, params)
    #motor_command = exploration ? actor_output.action : actor_output.mu
    motor_command = joystick_model(command)
    latent_ou_noise .= 0.98 * latent_ou_noise + 0.05 * CUDA.randn(60) * (1+joystick_noise)
    println(command)
    motor_command .+= latent_ou_noise

    decoder_input = cat(motor_command, prop; dims=1)
    reset_mask = CUDA.fill(false, 1)
    decoder_output = rollout!(actor_critic.decoder, decoder_input, false)
    mu, unscaled_sigma = split_halfway(decoder_output; dim=1)

    motor_action = Flux.cpu(mu)
    #env.last_torso_pos  = Environments.body_xpos(walker,  "walker/torso")
    #env.last_torso_quat = Environments.body_xquat(walker, "walker/torso")
    walker.data.ctrl .= clamp.(motor_action, -1.0, 1.0) |> Array
    for tt=1:params.physics.n_physics_steps
        Environments.step!(walker)
    end
    #MuJoCo.step!(model, data)
    #println(getindex(subtree_com(env, "walker/skull"), 3) * 10.0)
    #for tt=1:n_physics_steps
    #    MuJoCo.step!(env.model, env.data)
    #end
    #new_quat = body_xquat(env, "walker/torso")
    #env.lifetime += 1
    buttons = Bool.(GLFW.GetJoystickButtons(GLFW.JOYSTICK_1))
    if buttons[2]#status(env, params) != RUNNING || buttons[2]
        Environments.reset!(walker)
        Flux.reset!(actor_critic)
    end
end

MuJoCo.visualise!(walker.model, walker.data; controller=control!)


#while true
#    println(GLFW.GetJoystickAxes(GLFW.JOYSTICK_1))
#    sleep(.1)
#end 