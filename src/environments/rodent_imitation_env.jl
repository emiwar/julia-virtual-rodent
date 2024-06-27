include("mujoco_env.jl")
import LinearAlgebra

abstract type RodentFollowEnv <: MuJoCoEnv end

mutable struct RodentImitationEnv <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    lifetime::Int64
    cumulative_reward::Float64
    com_targets::Matrix{Float64}
    #torso::MuJoCo.Wrappers.NamedAccess.DataBody
end

function RodentImitationEnv()
    modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
    trajectoryPath = "src/environments/assets/example_com_trajectory.h5"
    com_targets = HDF5.h5open(fid->fid["com"][:, :], trajectoryPath, "r")
    #com_targets[3, :] .= 0.043
    model = MuJoCo.load_model(modelPath)
    data = MuJoCo.init_data(model)
    env = RodentImitationEnv(model, data, 0, 0.0, com_targets)
    reset!(env)
    return env
end

function clone(env::RodentImitationEnv)
    new_env = RodentImitationEnv(
        env.model,
        MuJoCo.init_data(env.model),
        0, 0.0,
        env.com_targets
    )
    reset!(new_env)
    return new_env
end

#Read-outs
function state(env::RodentFollowEnv, params)
    ComponentTensor(
     qpos=reshape(env.data.qpos, :),
     qvel=reshape(env.data.qvel, :),
     act=reshape(env.data.act, :),
     head_accel = read_sensor_value(env, "accelerometer") .* 0.1,
     head_vel = read_sensor_value(env, "velocimeter"),
     head_gyro = read_sensor_value(env, "gyro"),
     paw_contacts = [read_sensor_value(env, "palm_L"); 
                     read_sensor_value(env, "palm_R");
                     read_sensor_value(env, "sole_L");
                     read_sensor_value(env, "sole_R")],
     torso_linvel = read_sensor_value(env, "torso"),
     torso_xmat = MuJoCo.body(env.data, "torso").xmat,
     torso_height = MuJoCo.body(env.data, "torso").com[3] .* 10.0,
     com_target_array = reshape(get_future_targets(env, params), Val(1)) .* 5.0
    )
end

function reward(env::RodentFollowEnv, params)
    target_vec = get_target_vector(env, params)
    closeness_reward = exp(-sum(target_vec.^2) / params.reward_sigma_sqr)
    ctrl_reward = -params.ctrl_reward_weight * sum(env.data.ctrl.^2)
    return closeness_reward + ctrl_reward + params.healthy_reward_weight
    
end

function is_terminated(env::RodentFollowEnv, params)
    target_vec = get_target_vector(env, params)
    LinearAlgebra.norm(target_vec) > params.max_target_distance
end

function info(env::RodentFollowEnv)
    ComponentTensor(;
        torso_x=torso_x(env),
        torso_y=torso_y(env),
        torso_z=torso_z(env),
        lifetime=float(env.lifetime),
        cumulative_reward=env.cumulative_reward,
        actuator_force_sum_sqr=sum(env.data.actuator_force.^2),
        com_target_info(env, params)...
    )
end

#Actions
function act!(env::RodentFollowEnv, action, params)
    env.data.ctrl .= clamp.(action, -1.0, 1.0)
    for _=1:params.n_physics_steps
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    env.cumulative_reward += reward(env, params)
end

function reset!(env::RodentFollowEnv)
    MuJoCo.reset!(env.model, env.data)
    MuJoCo.forward!(env.model, env.data) #Run model forward to get correct initial state
    env.lifetime = 0
    env.cumulative_reward = 0.0
end

#Utils
torso_x(env::RodentFollowEnv) = MuJoCo.body(env.data, "torso").com[1]
torso_y(env::RodentFollowEnv) = MuJoCo.body(env.data, "torso").com[2]
torso_z(env::RodentFollowEnv) = MuJoCo.body(env.data, "torso").com[3]

#Targets
function get_future_targets(env::RodentImitationEnv, params)
    com_ind = env.lifetime ÷ 2
    range = (com_ind + 1):(com_ind + params.imitation_steps_ahead)
    view(env.com_targets, :, range) .- MuJoCo.body(env.data, "torso").com
end

function get_target_vector(env::RodentImitationEnv, params)
    com_ind = env.lifetime ÷ 2 + 1
    view(env.com_targets, :, com_ind) .- MuJoCo.body(env.data, "torso").com
end

function com_target_info(env::RodentFollowEnv, params)
    target_vec = get_target_vector(env, params)
    dist = LinearAlgebra.norm(target_vec)
    (target_vec_x=target_vec[1],
     target_vec_y=target_vec[2],
     target_vec_z=target_vec[3],
     target_distance=dist)
end


mutable struct RodentEightPathEnv <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    lifetime::Int64
    cumulative_reward::Float64
end

function RodentEightPathEnv()
    modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
    model = MuJoCo.load_model(modelPath)
    data = MuJoCo.init_data(model)
    env = RodentEightPathEnv(model, data, 0, 0.0)
    reset!(env)
    return env
end

function clone(env::RodentEightPathEnv)
    new_env = RodentEightPathEnv(
        env.model,
        MuJoCo.init_data(env.model),
        0, 0.0)
    reset!(new_env)
    return new_env
end

target_speed(env::RodentEightPathEnv) = 0.2
radius(env::RodentEightPathEnv) = 1.0

function get_future_targets(env::RodentEightPathEnv, params)
    factor = 2pi * radius(env) / target_speed(env) / env.model.opt.timestep / params.n_physics_steps
    t = (env.lifetime:(env.lifetime+params.imitation_steps_ahead)) ./ factor
    com = MuJoCo.body(env.data, "torso").com
    vcat(radius(env).*sin.(2.0 .* 2pi .* t') .- com[1],
         radius(env).*(cos.(2pi .* t') .- 1.0) .- com[2],
         zero(t'))
end

function get_target_vector(env::RodentEightPathEnv, params)
    factor = 2pi / target_speed(env) / env.model.opt.timestep / params.n_physics_steps
    t = (env.lifetime-1) / factor
    com = MuJoCo.body(env.data, "torso").com
    [radius(env)*sin(2.0*2pi*t) - com[1], radius(env)*(cos(2pi*t)-1.0) - com[2], 0.0]
end

