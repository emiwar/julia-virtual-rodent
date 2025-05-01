import Wandb
import BSON
import Dates
import TOML
import PythonCall
import CUDA
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

wandb_run_id = "zf8zs3kq" #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
falloffs = params.reward.falloff
falloffs = (;com=falloffs.com, rotation=falloffs.rotation, joint=falloffs.joint, joint_vel=falloffs.joint_vel, appendages=falloffs.appendages)
reward_params = merge(params.reward, (;falloff=falloffs))

actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

#Setup the environment
walker = Environments.Rodent(;merge(params.physics, (;body_scale=1.0))...)
reward_spec = Environments.EqualRewardWeights(;reward_params...)
target = Environments.load_imitation_target(walker)
template_env = Environments.ImitationEnv(walker, reward_spec, target; params.imitation...)
n_envs = size(target)[3]
env = Environments.MultithreadEnv(template_env, n_envs)
batch_dims = (500, n_envs)

statuses = fill(-1, batch_dims)
latents = fill(-1.0, (params.network.latent_dimension, 500, n_envs))

head_heights = fill(NaN, batch_dims)
timestep::Float64 = template_env.walker.model.opt.timestep * params.physics.n_physics_steps

torso_xpos = Vector{SVector{3, Float64}}[]
torso_xquat = Vector{SVector{4, Float64}}[]
torso_xmat = Vector{SMatrix{3, 3, Float64}}[]
head_heights = Vector{Float64}[]

for (i, env) in enumerate(env.environments)
    Environments.reset!(env, i, 1)
end
Flux.reset!(actor_critic)
Environments.prepare_epoch!(env)
ProgressMeter.@showprogress for t=1:batch_dims[1]
    reset_mask = Environments.status(env) .!= Environments.RUNNING
    #
    imitation_target = Environments.state(env).imitation_target |> array
    proprioception = Environments.state(env).proprioception |> array
    latent = rollout!(actor_critic.encoder, imitation_target |> gpu, reset_mask |> gpu)
    decoder_input = cat(latent, proprioception |> gpu; dims=1)
    decoder_output = rollout!(actor_critic.decoder, decoder_input, reset_mask |> gpu)
    mu, unscaled_sigma = split_halfway(decoder_output; dim=1)
    Environments.act!(env, mu)
    all_walkers = map(e->e.walker, env.environments)
    push!(torso_xpos,  Environments.body_xpos.(all_walkers, "walker/torso"))
    push!(torso_xquat, Environments.body_xquat.(all_walkers, "walker/torso"))
    push!(torso_xmat,  Environments.body_xmat.(all_walkers, "walker/torso"))
    push!(head_heights, getindex.(Environments.subtree_com.(all_walkers, "walker/skull"), 3))
    statuses[t, :] = Environments.status(env)
    latents[:, t, :] = latent |> Flux.cpu
end

commands = Vector{Float64}[]
commands_latents = Vector{Float64}[]
for t = 1:batch_dims[1]-3
    for i = 1:n_envs
        if any(statuses[t:t+3, i] .!= 0)
            continue
        end
        current_xpos = torso_xpos[t][i]
        current_xmat = torso_xmat[t][i]
        current_xquat = torso_xquat[t][i]
        future_xpos = torso_xpos[t+3][i]
        future_xquat = torso_xquat[t+3][i]
        forward_speed_c = (current_xmat * (future_xpos - current_xpos))[1] / (3*timestep)
        turning_speed_c = Environments.azimuth_between(current_xquat, future_xquat) / (3*timestep)
        head_height_c = head_heights[t+3][i] * 10.0
        push!(commands, [forward_speed_c, turning_speed_c, head_height_c])
        push!(commands_latents, latents[:, t, i])
    end
end
traininputs = fill(NaN, (3, length(commands)))
trainoutputs = fill(NaN, (60, length(commands)))
for t=eachindex(commands)
    traininputs[:, t] = commands[t]
    trainoutputs[:, t] = commands_latents[t]
end

model = Chain(Dense(3 => 1024, tanh), Dense(1024 => 1024, tanh), Dense(1024=>60)) #Dense(1024=>1024, tanh),

traininputs = gpu(traininputs)
trainoutputs = gpu(trainoutputs)
model = gpu(model)

#model(traininputs)

loss(model, x, y) = sum((model(x) .- y).^2)
#opt = Adam()
opt_state = Flux.setup(Adam(), model)

dataloader = Flux.DataLoader((traininputs, trainoutputs); batchsize=1024, shuffle=true)
losses = Float64[]
@showprogress for t=1:25
    push!(losses, loss(model, traininputs, trainoutputs))
    for batch in dataloader
        Flux.train!(loss, model, [batch], opt_state)
    end
end



joystick_model = Flux.cpu(model)

BSON.bson("joystick_model5.bson"; joystick_model)

import Plots
Plots.plot(losses)
forward_speeds = traininputs[1, :]
ex_data = map(x->x>0.35 && x<0.45, forward_speeds)
trainoutputs[:, ex_data]
trainoutputs[:, ex_data]' * model(CUDA.cu([0.0, 0.0, 0.06]))

mean_output = Statistics.mean(trainoutputs[:, ex_data], dims=2)[:, 1]
output_of_mean = model(gpu([0.5, 0.0, 0.0]))
Statistics.cor(mean_output |> Flux.cpu, output_of_mean |> Flux.cpu)