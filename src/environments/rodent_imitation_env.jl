include("mujoco_env.jl")
include("imitation_trajectory.jl")
import LinearAlgebra: norm

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
    if params.physics.torque_control
        modelPath = "src/environments/assets/rodent_with_floor_scale080_torques.xml"
    else
        modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
    end
    model = MuJoCo.load_model(modelPath)
    if params.physics.iterations !== nothing
        unsafe_store!(Ptr{Int32}(model.opt.internal_pointer + 264), Int32(params.physics.iterations))
        @assert model.opt.iterations == params.physics.iterations
        #model.opt.iterations = params.physics.iterations
    end
    if params.physics.ls_iterations !== nothing
        unsafe_store!(Ptr{Int32}(model.opt.internal_pointer + 268), Int32(params.physics.ls_iterations))
        @assert model.opt.ls_iterations == params.physics.ls_iterations
        #model.opt.ls_iterations = params.physics.ls_iterations
    end
    data = MuJoCo.init_data(model)
    target = ImitationTarget(model)
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
     com_target_array = reshape(future_targets(env, params), :),
     xquat_target_array = reshape(future_quats(env, params), :),
     joint_target_array = reshape(future_joint_pos(env, params), :),
     appendages_target_array = reshape(future_appendages_pos(env, params), :)
    )
end

function reward(env::RodentFollowEnv, params)
    target_vec = target_vector(env, params)
    com_reward = exp(-sum(target_vec.^2) / (params.reward.falloff.com^2))
    angle_reward = exp(-(angle_to_target(env, params)^2) / (params.reward.falloff.rotation^2))
    joint_reward = exp(-sum(joint_error(env, params).^2) / (params.reward.falloff.joint^2))
    append_reward = appendages_reward(env, params)
    ctrl_reward = -params.reward.control_cost * sum(env.data.ctrl.^2)
    total_reward  = com_reward + angle_reward + joint_reward + append_reward
    total_reward += ctrl_reward + params.reward.alive_bonus
    total_reward #return clamp(total_reward, params.min_reward, Inf)
end

function status(env::RodentFollowEnv, params)
    if torso_z(env) < params.physics.min_torso_z 
        return TERMINATED
    elseif target_frame(env) + params.imitation.horizon >= length(env.target)
        return TRUNCATED
    elseif norm(target_vector(env, params)) > params.imitation.max_target_distance
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
        joint_reward = exp(-sum(joint_error(env, params).^2) / (params.reward.falloff.joint)^2),
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
    dist = norm(target_vec)
    app_error = appendages_error(env, params)
    spawn_point = view(env.target.com, :, 1, env.target_clip)
    com = MuJoCo.body(env.data, "torso").com
    (target_frame=target_frame(env),
     target_vec_x=target_vec[1],
     target_vec_y=target_vec[2],
     target_vec_z=target_vec[3],
     target_distance=dist,
     distance_from_spawpoint=norm(spawn_point .- com),
     joint_error=norm(joint_error(env, params)),
     lower_arm_R_error=norm(view(app_error, :, 1)),
     lower_arm_L_error=norm(view(app_error, :, 2)),
     foot_R_error=norm(view(app_error, :, 3)),
     foot_L_error=norm(view(app_error, :, 4)),
     jaw_error=norm(view(app_error, :, 5))
     )
end

function imitation_horizon(env::RodentImitationEnv, params)
    (target_frame(env) + 1):(target_frame(env) + params.imitation.horizon)
end

function future_quats(env::RodentImitationEnv, params)
    rot_vec = zeros(3, params.imitation.horizon)
    diff = zeros(3)
    for (i, t) in enumerate(imitation_horizon(env, params))
        MuJoCo.mju_subQuat(diff, env.target.qpos[4:7, t, env.target_clip], env.data.qpos[4:7])
        rot_vec[:, i] .= diff
    end
    return rot_vec
end

function angle_to_target(env::RodentImitationEnv, params)
    target_quat = view(env.target.qpos, 4:7, target_frame(env), env.target_clip)
    current_quat = MuJoCo.body(env.data, "torso").xquat
    quat_prod = target_quat' * current_quat
    return 2*acos(abs(clamp(quat_prod, -1, 1))) #Concerning that clamp is needed here
end

function future_joint_pos(env::RodentImitationEnv, params)
    joint_indices = 8:size(env.target.qpos, 1)
    view(env.target.qpos, joint_indices, imitation_horizon(env, params), env.target_clip)# .- env.data.qpos[joint_indices]
end

function joint_error(env::RodentImitationEnv, params)
    joint_indices = 8:size(env.target.qpos, 1)
    view(env.target.qpos, joint_indices, target_frame(env), env.target_clip) .- env.data.qpos[joint_indices]
end

function future_appendages_pos(env::RodentImitationEnv, params)
    view(env.target.appendages, :, :, imitation_horizon(env, params), env.target_clip)
end

function appendages_pos(env::RodentImitationEnv, params)
    torso = MuJoCo.body(env.data, "torso")
    torso_xmat = reshape(torso.xmat, 3, 3)
    positions = zeros(3, length(appendages()))
    for (i, app_name) in enumerate(appendages())
        app_body = MuJoCo.body(env.data, app_name)
        positions[:, i] = torso_xmat*(app_body.xpos .- torso.xpos)
    end
    return positions
end

function appendages_error(env::RodentImitationEnv, params)
    view(env.target.appendages, :, :, target_frame(env), env.target_clip) .- appendages_pos(env, params)
end

function appendages_reward(env::RodentImitationEnv, params)
    app_error = appendages_error(env, params)
    sum(exp(-sum(view(app_error, :, i).^2) / (params.reward.falloff.appendages^2)) for i=1:5) / 5.0
end

