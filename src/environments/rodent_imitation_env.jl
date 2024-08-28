include("mujoco_env.jl")
include("imitation_trajectory.jl")
import LinearAlgebra

abstract type RodentFollowEnv <: MuJoCoEnv end

mutable struct RodentImitationEnv <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    target::ImitationTarget
    target_frame::Int64
    target_clip::Int64
    lifetime::Int64
    cumulative_reward::Float64
    #torso::MuJoCo.Wrappers.NamedAccess.DataBody
end

function RodentImitationEnv(params)
    modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
    model = MuJoCo.load_model(modelPath)
    data = MuJoCo.init_data(model)
    target = ImitationTarget()
    env = RodentImitationEnv(model, data, target, 0, 1, 0, 0.0)
    reset!(env, params)
    return env
end

function clone(env::RodentImitationEnv, params)
    new_env = RodentImitationEnv(
        env.model,
        MuJoCo.init_data(env.model),
        env.target,
        0,1,0,0.0
    )
    reset!(new_env, params)
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
     com_target_array = reshape(future_targets(env, params), Val(1)) .* 5.0,
     xquat_target_array = reshape(future_quats(env, params), Val(1))
    )
end

function reward(env::RodentFollowEnv, params)
    target_vec = target_vector(env, params)
    target_vec[3] *= 0.2 #Downplay importance of rearing
    closeness_reward = exp(-sum(target_vec.^2) / params.reward_sigma_sqr)
    angle_reward = exp(-(angle_to_target(env, params)^2) / params.reward_angle_sigma_sqr)
    ctrl_reward = -params.ctrl_reward_weight * sum(env.data.ctrl.^2)
    total_reward = closeness_reward + angle_reward + ctrl_reward + params.healthy_reward_weight
    return clamp(total_reward, params.min_reward, Inf)
end

function status(env::RodentFollowEnv, params)
    if torso_z(env) < params.min_torso_z 
        return TERMINATED
    elseif target_frame(env) + params.imitation_steps_ahead > 245
        return TRUNCATED
    elseif LinearAlgebra.norm(target_vector(env, params)) > params.max_target_distance
        return TERMINATED
    else
        return RUNNING
    end
end

function info(env::RodentFollowEnv)
    ComponentTensor(;
        torso_x=torso_x(env),
        torso_y=torso_y(env),
        torso_z=torso_z(env),
        lifetime=float(env.lifetime),
        cumulative_reward=env.cumulative_reward,
        actuator_force_sum_sqr=sum(env.data.actuator_force.^2),
        angle_to_target=angle_to_target(env, params) |> rad2deg,
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
    if env.lifetime % 2 == 0 #Hack since target update each 20ms but simulation each 10ms (but this depends on params so shouldn't be hard-coded here)
        env.target_frame += 1
    end
    env.cumulative_reward += reward(env, params)
end

function reset!(env::RodentFollowEnv, params)
    env.lifetime = 0
    env.cumulative_reward = 0.0
    env.target_clip = rand(1:size(env.target.qpos, 1))
    env.target_frame = 1

    MuJoCo.reset!(env.model, env.data)
    env.data.qpos .= view(env.target.qpos, :, target_frame(env), env.target_clip)
    env.data.qpos[3] += params.spawn_z_offset
    env.data.qvel .= view(env.target.qvel, :, target_frame(env), env.target_clip)
    MuJoCo.forward!(env.model, env.data) #Run model forward to get correct initial state
end

#Utils
torso_x(env::RodentFollowEnv) = MuJoCo.body(env.data, "torso").com[1]
torso_y(env::RodentFollowEnv) = MuJoCo.body(env.data, "torso").com[2]
torso_z(env::RodentFollowEnv) = MuJoCo.body(env.data, "torso").com[3]

#Targets
target_frame(env::RodentImitationEnv) = env.target_frame
function future_targets(env::RodentImitationEnv, params)
    current_com = MuJoCo.body(env.data, "torso").com
    current_xmat = reshape(MuJoCo.body(env.data, "torso").xmat, 3, 3)
    current_xmat * (view(env.target.com, :, imitation_horizon(env, params), env.target_clip) .- current_com)
end

function target_vector(env::RodentImitationEnv, params)
    view(env.target.com, :, target_frame(env), env.target_clip) .- MuJoCo.body(env.data, "torso").com
end

function com_target_info(env::RodentFollowEnv, params)
    target_vec = target_vector(env, params)
    target_vec[3] *= 0.2 #Downplay importance of rearing
    dist = LinearAlgebra.norm(target_vec)
    (target_frame=target_frame(env),
     target_vec_x=target_vec[1],
     target_vec_y=target_vec[2],
     target_vec_z=target_vec[3],
     target_distance=dist,
     distance_from_spawpoint=LinearAlgebra.norm(view(env.target.com, :, 1, env.target_clip) .- MuJoCo.body(env.data, "torso").com))
end

function imitation_horizon(env::RodentImitationEnv, params)
    (target_frame(env) + 1):(target_frame(env) + params.imitation_steps_ahead)
end

function future_quats(env::RodentImitationEnv, params)
    rot_vec = zeros(3, params.imitation_steps_ahead)
    #target_qpos = view(env.target.qpos, 4:7, imitation_horizon(env, params), env.target_clip)
    diff = zeros(3)
    for (i, t) in enumerate(imitation_horizon(env, params))
        MuJoCo.mju_subQuat(diff, env.target.qpos[4:7, t, env.target_clip], env.data.qpos[4:7])
        rot_vec[:, i] .= diff
    end
    return rot_vec
end

#function future_xmats(env::RodentImitationEnv, params)
#    view(env.target.xmat, :, imitation_horizon(env, params))
#end

function angle_to_target(env::RodentImitationEnv, params)
    target_quat = view(env.target.qpos, 4:7, target_frame(env), env.target_clip)
    current_quat = MuJoCo.body(env.data, "torso").xquat
    quat_prod = target_quat' * current_quat
    return 2*acos(abs(clamp(quat_prod, -1, 1))) #Concerning that clamp is needed here
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

