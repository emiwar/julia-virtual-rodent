import Wandb
import BSON
import Dates
import TOML
import PythonCall
import CUDA
import MuJoCo
using StaticArrays: SVector, SMatrix
using ProgressMeter
import Flux; using Flux: Chain, gpu, Dense, Adam
using LinearAlgebra: norm, dot
include("../src/utils/parse_config.jl")
include("../src/utils/component_tensor.jl")
include("../src/utils/profiler.jl")
include("../src/utils/wandb_logger.jl")
include("../src/environments/environments.jl")
include("../src/networks/networks.jl")
include("../src/networks/utils.jl")
include("../src/algorithms/algorithms.jl")

using .Networks: EncDec, VariationalBottleneck, GaussianActionSampler#, split_halfway
using .ComponentTensors: ComponentTensor, array

wandb_run_id = "zf8zs3kq" #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu
joystick_model = BSON.load("joystick_model5.bson")[:joystick_model] |> Flux.gpu

#Setup the environment
walker = Environments.Rodent(;merge(params.physics, (;body_scale=1.0))...)
Environments.reset!(walker)
Flux.reset!(actor_critic)

MuJoCo.init_visualiser()
exploration = false
latent_ou_noise = CUDA.zeros(60)

struct JoystickState
    axes::Vector{Float64}
    buttons::Vector{UInt8}
    lock::ReentrantLock
end
const joystick_state = JoystickState(zeros(Float64, 6), zeros(UInt8, 15), ReentrantLock())
read_axes(js::JoystickState, ind) = @lock js.lock js.axes[ind]
read_buttons(js::JoystickState, ind) = @lock js.lock js.buttons[ind]
function update!(js::JoystickState)
    lock(js.lock)
    try
        js.axes .= GLFW.GetJoystickAxes(GLFW.JOYSTICK_1)
        js.buttons .= GLFW.GetJoystickButtons(GLFW.JOYSTICK_1)
    finally
        unlock(js.lock)
    end
end
@async begin
    while true
        update!(joystick_state)
        sleep(0.01)
    end
end

const physics_steps = Ref(0)

function control!(model, data)
    if physics_steps[] >= 5
        prop = Environments.proprioception(walker) |> ComponentTensor |> ComponentTensors.array |> gpu
        joystick_noise, joystick_turn, joystick_backwards, joystick_rear = read_axes(joystick_state, 3:6)
        joystick_forward = -0.9joystick_backwards#clamp(-0.5joystick_backwards, -.6, 1.0)
        command = Flux.gpu([joystick_forward, -3joystick_turn, 1.2 + 0.6*joystick_rear])
        motor_command = joystick_model(command)
        latent_ou_noise .= 0.98 * latent_ou_noise + 0.05 * CUDA.randn(60) * (1+joystick_noise)
        motor_command .+= latent_ou_noise

        decoder_input = cat(motor_command, prop; dims=1)
        reset_mask = CUDA.fill(false, 1)
        decoder_output = rollout!(actor_critic.decoder, decoder_input, false)
        mu, unscaled_sigma = split_halfway(decoder_output; dim=1)

        motor_action = Flux.cpu(mu)
        walker.data.ctrl .= clamp.(motor_action, -1.0, 1.0) |> Array
        physics_steps[] = 0
    end
    if Bool(read_buttons(joystick_state, 2))#status(env, params) != RUNNING || buttons[2]
        Environments.reset!(walker)
        Flux.reset!(actor_critic)
    end
    physics_steps[] += 1
end

MuJoCo.visualise!(walker.model, walker.data; controller=control!)
