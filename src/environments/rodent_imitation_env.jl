include("mujoco_env.jl")
include("imitation_trajectory.jl")
include("../utils/mujoco_quat.jl")
include("../utils/load_dm_control_model.jl")
import LinearAlgebra: norm
using StaticArrays
abstract type RodentFollowEnv <: MuJoCoEnv end

mutable struct RodentImitationEnv{ImLen} <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    target::ImitationTarget
    target_frame::Int64
    target_clip::Int64
    lifetime::Int64
    cumulative_reward::Float64
    sensorranges::Dict{String, UnitRange{Int64}}
end

function RodentImitationEnv(params)
    model = dm_control_rodent(torque_actuators = params.physics.torque_control,
                              foot_mods = params.physics.foot_mods,
                              scale = params.physics.body_scale,
                              physics_timestep = params.physics.timestep,
                              control_timestep = params.physics.timestep * params.physics.n_physics_steps)
    data = MuJoCo.init_data(model)
    target = ImitationTarget(model)
    sensorranges = prepare_sensorranges(model, "walker/" .* ["accelerometer", "velocimeter",
                                                             "gyro", "palm_L", "palm_R",
                                                             "sole_L", "sole_R", "torso"])
    env = RodentImitationEnv{params.imitation.horizon}(model, data, target, 0, 1, 0, 0.0, sensorranges)
    reset!(env, params)
    return env
end

function clone(env::RodentImitationEnv{ImLen}, params) where ImLen
    new_env = RodentImitationEnv{ImLen}(
        env.model,
        MuJoCo.init_data(env.model),
        env.target,
        0,1,0,0.0,env.sensorranges
    )
    reset!(new_env, params)
    return new_env
end

function state(env::RodentImitationEnv, params)
    (
        proprioception = (
            joints = (@view env.data.qpos[8:end]),
            joint_vel = (@view env.data.qvel[7:end]),
            actuations = (@view env.data.act[:]),
            head = (
                velocity = sensor(env, "walker/velocimeter"),
                accel = sensor(env, "walker/accelerometer"),
                gyro = sensor(env, "walker/gyro")
            ),
            torso = (
                velocity = sensor(env, "walker/torso"),
                xmat = reshape(body_xmat(env, "walker/torso"), :),
                com = subtree_com(env, "walker/torso")
            ),
            paw_contacts = (
                palm_L = sensor(env, "walker/palm_L"),
                palm_R = sensor(env, "walker/palm_R"),
                sole_L = sensor(env, "walker/sole_L"),
                sole_R = sensor(env, "walker/sole_R")
            )
        ),
        imitation_target = (
            com = reshape(com_horizon(env), :),
            root_quat = reshape(root_quat_horizon(env), :),
            joints = reshape(joints_horizon(env), :),
            appendages = reshape(appendages_pos_horizon(env), :)
        )
    )
end

#Read-outs
function reward(env::RodentFollowEnv, params)
    target_vec = com_error(env)
    com_reward = exp(-(norm(target_vec)^2) / (params.reward.falloff.com^2))
    angle_reward = exp(-(angle_to_target(env)^2) / (params.reward.falloff.rotation^2))
    joint_reward = exp(-joint_error(env) / (params.reward.falloff.joint^2))
    append_reward = appendages_reward(env, params)
    ctrl_reward = -params.reward.control_cost * norm(env.data.ctrl)^2
    total_reward  = com_reward + angle_reward + joint_reward + append_reward
    total_reward += ctrl_reward + params.reward.alive_bonus
    total_reward #return clamp(total_reward, params.min_reward, Inf)
end

function status(env::RodentFollowEnv, params)
    target_distance = norm(com_error(env))
    if torso_z(env) < params.physics.min_torso_z 
        return TERMINATED
    elseif target_frame(env) + params.imitation.horizon >= length(env.target)
        return TRUNCATED
    elseif target_distance > params.imitation.max_target_distance
        return TERMINATED
    else
        return RUNNING
    end
end

function info(env::RodentFollowEnv, params)
    (
        torso_x=torso_x(env),
        torso_y=torso_y(env),
        torso_z=torso_z(env),
        lifetime=float(env.lifetime),
        cumulative_reward=env.cumulative_reward,
        actuator_force_sum_sqr=norm(env.data.actuator_force)^2,
        angle_to_target=angle_to_target(env) |> rad2deg,
        joint_reward = exp(-sum(joint_error(env).^2) / (params.reward.falloff.joint)^2),
        appendages_reward = appendages_reward(env, params),
        com_target_info(env, params)...
    )
end

#Actions
function act!(env::RodentFollowEnv, action, params)
    env.data.ctrl .= clamp.(action, -1.0, 1.0)
    for _=1:params.physics.n_physics_steps
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
    env.data.qpos[3] += params.physics.spawn_z_offset
    env.data.qvel .= view(env.target.qvel, :, target_frame(env), env.target_clip)
    MuJoCo.forward!(env.model, env.data) #Run model forward to get correct initial state
end

#Utils
torso_x(env::RodentFollowEnv) = subtree_com(env, "walker/torso")[1]
torso_y(env::RodentFollowEnv) = subtree_com(env, "walker/torso")[2]
torso_z(env::RodentFollowEnv) = subtree_com(env, "walker/torso")[3]

#Targets
target_frame(env::RodentImitationEnv) = env.target_frame
function imitation_horizon(env::RodentImitationEnv{N}) where N
    (target_frame(env) + 1):(target_frame(env) + N)
end

#Center-of-mass target
target_com(env::RodentImitationEnv) = target_com(env, target_frame(env))
target_com(env::RodentImitationEnv, t) = SVector{3}(view(env.target.com, :, t, env.target_clip))
function relative_com(env::RodentImitationEnv, t::Int64)
    body_xmat(env, "walker/torso") * (target_com(env, t) - subtree_com(env, "walker/torso"))
end
@generated function com_horizon(env::RodentImitationEnv{N}) where N
    return :(hcat($((:(relative_com(env, target_frame(env) + $i)) for i=1:N)...)))
end
com_error(env::RodentImitationEnv) = target_com(env) - subtree_com(env, "walker/torso")

#Root quaternion target
target_root_quat(env::RodentImitationEnv) = target_root_quat(env, target_frame(env))
target_root_quat(env::RodentImitationEnv, t) = SVector{4}(view(env.target.qpos, 4:7, t, env.target_clip))
function relative_root_quat(env::RodentImitationEnv, t::Int64)::SVector{3, Float64}
    subQuat(target_root_quat(env, t), body_xquat(env, "walker/torso"))
end
@generated function root_quat_horizon(env::RodentImitationEnv{N}) where N
    return :(hcat($((:(relative_root_quat(env, target_frame(env) + $i)) for i=1:N)...)))
end
function angle_to_target(env::RodentImitationEnv)
    quat_prod = target_root_quat(env)' * body_xquat(env, "walker/torso")
    return 2*acos(abs(clamp(quat_prod, -1, 1))) #Concerning that clamp is needed here
end

#Joints
function target_joints(env::RodentImitationEnv, t)
    @view env.target.qpos[8:end, t, env.target_clip]
end
function joints_horizon(env::RodentImitationEnv{N}) where N
    @view env.target.qpos[8:end, imitation_horizon(env), env.target_clip]
end
function joint_error(env::RodentImitationEnv)
    joint_indices = 8:size(env.target.qpos, 1)
    target_joint = view(env.target.qpos, :, target_frame(env), env.target_clip)
    err = 0.0
    for ji in joint_indices
        err += (target_joint[ji] - env.data.qpos[ji])^2
    end
    return err
end

#Appendages
function appendage_pos(env::RodentImitationEnv, appendage_name::String)
    body_xmat(env, "walker/torso")*(body_xpos(env, appendage_name) .- body_xpos(env, "walker/torso"))
end
@generated function appendages_pos(env::RodentImitationEnv)
    :(hcat($((:(appendage_pos(env, $a)) for a=appendages())...)))
end
appendages_pos_horizon(env) = appendages_pos_horizon(env, target_frame(env))
function appendages_pos_horizon(env, t)
    SMatrix{3, length(appendages())}(view(env.target.appendages, :, :, t, env.target_clip))
end
function appendages_error(env::RodentImitationEnv)
    appendages_pos_horizon(env) - appendages_pos(env)
end
function appendages_reward(env::RodentImitationEnv, params)
    reward = 0.0
    errors = appendages_error(env)
    for i = axes(errors, 2)
        reward += exp(-norm(view(errors, :, i))^2 / (params.reward.falloff.appendages^2))
    end
    return reward / length(appendages())
end

function com_target_info(env::RodentFollowEnv, params)
    target_vec = target_com(env) - subtree_com(env, "walker/torso")
    dist = norm(target_vec)
    app_error = appendages_error(env)
    spawn_point = target_com(env, 1)
    (target_frame=target_frame(env),
     target_vec_x=target_vec[1],
     target_vec_y=target_vec[2],
     target_vec_z=target_vec[3],
     target_distance=dist,
     distance_from_spawpoint=norm(spawn_point - subtree_com(env, "walker/torso")),
     joint_error=sqrt(joint_error(env)),
     lower_arm_R_error=norm(view(app_error, :, 1)),
     lower_arm_L_error=norm(view(app_error, :, 2)),
     foot_R_error=norm(view(app_error, :, 3)),
     foot_L_error=norm(view(app_error, :, 4)),
     jaw_error=norm(view(app_error, :, 5))
     )
end