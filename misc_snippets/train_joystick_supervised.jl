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
include("../src/environments/imitation_trajectory.jl")
include("../src/environments/rodent_imitation_env.jl")
include("../src/networks/variational_enc_dec.jl")
include("../src/collectors/batch_stepper.jl")

exploration = false
wandb_run_id = "j0zwbgns" #"7mzfglak"

params, weights_file_name = load_from_wandb(wandb_run_id, r"step-.*")
ActorCritic = VariationalEncDec
actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu

template_env = RodentImitationEnv(params)#, target_data="reference_data/2020_12_22_1_precomputed.h5")
stepper = BatchStepper(template_env, size(template_env.target)[3])
for (i, env) in enumerate(stepper.environments)
    reset!(env, params, i, 1)
end
batch_dims = (500, size(template_env.target)[3])
statuses = fill(-1, batch_dims)
latents = fill(-1.0, (params.network.latent_dimension, 500, size(template_env.target)[3]))

head_heights = fill(NaN, batch_dims)
timestep::Float64 = template_env.model.opt.timestep * params.physics.n_physics_steps
cu(ct::ComponentTensor) = ComponentTensor(CUDA.cu(data(ct)), index(ct))

torso_xpos = Vector{SVector{3, Float64}}[]
torso_xquat = Vector{SVector{4, Float64}}[]
torso_xmat = Vector{SMatrix{3, 3, Float64}}[]
head_heights = Vector{Float64}[]
for (i, env) in enumerate(stepper.environments)
    reset!(env, params, i, 1)
end
prepareEpoch!(stepper, params)
ProgressMeter.@showprogress for t=1:batch_dims[1]
    actor_output = actor(actor_critic, cu(stepper.states), params)
    copyto!(stepper.actions, exploration ? actor_output.action : actor_output.mu)
    step!(stepper, params)
    push!(torso_xpos, body_xpos.(stepper.environments, "walker/torso"))
    push!(torso_xquat, body_xquat.(stepper.environments, "walker/torso"))
    push!(torso_xmat, body_xmat.(stepper.environments, "walker/torso"))
    push!(head_heights, getindex.(subtree_com.(stepper.environments, "walker/skull"), 3))
    statuses[t, :] = stepper.status
    latents[:, t, :] = Array(actor_output.latent_mu)
end

commands = Vector{Float64}[]
commands_latents = Vector{Float64}[]
for t = 1:batch_dims[1]-3
    for i = 1:length(stepper.environments)
        if any(statuses[t:t+3, i] .!= 0)
            continue
        end
        current_xpos = torso_xpos[t][i]
        current_xmat = torso_xmat[t][i]
        current_xquat = torso_xquat[t][i]
        future_xpos = torso_xpos[t+3][i]
        future_xquat = torso_xquat[t+3][i]
        forward_speed_c = (current_xmat * (future_xpos - current_xpos))[1] / (3*timestep)
        turning_speed_c = azimuth_between(current_xquat, future_xquat) / (3*timestep)
        head_height_c = head_heights[t+1][i] * 10.0
        push!(commands, [forward_speed_c, turning_speed_c, head_height_c])
        push!(commands_latents, latents[:, t, i])
    end
end
traininputs = fill(NaN, (3, length(commands)))
trainoutputs = fill(NaN, (60, length(commands)))
for t=1:length(commands)
    traininputs[:, t] = commands[t]
    trainoutputs[:, t] = commands_latents[t]
end

#=
ProgressMeter.@showprogress for t=1:batch_dims[1]-3
    actor_output = actor(actor_critic, cu(stepper.states), params)
    copyto!(stepper.actions, exploration ? actor_output.action : actor_output.mu)
    
    xmat = body_xmat.(stepper.environments, "walker/torso")
    last_torso_pos = body_xpos.(stepper.environments, "walker/torso")
    last_torso_quat = body_xquat.(stepper.environments, "walker/torso")
    step!(stepper, params)
    new_torso_pos = body_xpos.(stepper.environments, "walker/torso")
    new_torso_quat = body_xquat.(stepper.environments, "walker/torso")
    forward_speeds[t, :] = getindex.(xmat .* (new_torso_pos .- last_torso_pos), 1) ./ timestep
    turning_speeds[t, :] = azimuth_between.(last_torso_quat, new_torso_quat) ./ timestep
    head_heights[t, :] = getindex.(subtree_com.(stepper.environments, "walker/skull"), 3) .* 10.0
    #infos[:, t, :] = stepper.infos
    statuses[t, :] = stepper.status
    latents[:, t, :] = Array(actor_output.latent_mu)
end
=#
#forward_speeds = fill(NaN, batch_dims)
#turning_speeds = fill(NaN, batch_dims)

#mask = statuses[:] .== 0
#traininputs = hcat(forward_speeds[mask], turning_speeds[mask], head_heights[mask])' |> collect
#trainoutputs = reshape(latents, Val(2))[:, mask]
model = Chain(Dense(3 => 1024, tanh), Dense(1024=>1024, tanh), Dense(1024=>60))

traininputs = gpu(traininputs)
trainoutputs = gpu(trainoutputs)
model = gpu(model)

#model(traininputs)

loss(model, x, y) = sum((model(x) .- y).^2)
#opt = Adam()
opt_state = Flux.setup(Adam(), model)

dataloader = Flux.Data.DataLoader((traininputs, trainoutputs); batchsize=1024, shuffle=true)
losses = Float64[]
@showprogress for t=1:25
    push!(losses, loss(model, traininputs, trainoutputs))
    for batch in dataloader
        Flux.train!(loss, model, [batch], opt_state)
    end
end
#Plots.plot(losses)

#ex_data = map(x->x>0.35 && x<0.45, forward_speeds[mask])
#trainoutputs[:, ex_data]

#trainoutputs[:, ex_data]' * model(CUDA.cu([0.0, 0.0, 0.06]))

joystick_model = model |> cpu

BSON.bson("joystick_model3.bson"; joystick_model)