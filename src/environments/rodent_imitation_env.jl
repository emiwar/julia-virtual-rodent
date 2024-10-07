include("mujoco_env.jl")
include("imitation_trajectory.jl")
import LinearAlgebra: norm
using StaticArrays
abstract type RodentFollowEnv <: MuJoCoEnv end

mutable struct RodentImitationEnv <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    target::ImitationTarget
    target_frame::Int64
    target_clip::Int64
    lifetime::Int64
    cumulative_reward::Float64
    sensorranges::Dict{String, UnitRange{Int64}}
    #torso::MuJoCo.Wrappers.NamedAccess.DataBody
end

function RodentImitationEnv(params)
    if params.physics.torque_control
        modelPath = "src/environments/assets/rodent_with_floor_scale080_torques.xml"
    else
        modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
    end
    model = MuJoCo.load_model(modelPath)
    data = MuJoCo.init_data(model)
    target = ImitationTarget(model)
    sensorranges = prepare_sensorranges(model, ["accelerometer", "velocimeter", "gyro",
                                                "palm_L", "palm_R", "sole_L", "sole_R", "torso"])
    env = RodentImitationEnv(model, data, target, 0, 1, 0, 0.0, sensorranges)
    reset!(env, params)
    return env
end

function clone(env::RodentImitationEnv, params)
    new_env = RodentImitationEnv(
        env.model,
        MuJoCo.init_data(env.model),
        env.target,
        0,1,0,0.0,env.sensorranges
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

function state!(s, env::RodentFollowEnv, params)
    s.qpos .= reshape(env.data.qpos, :)
    s.qvel .= reshape(env.data.qvel, :)
    s.act .= reshape(env.data.act, :)
    s.head_accel .= view(env.data.sensordata, env.sensorranges["accelerometer"]) .* 0.1
    s.head_vel .= view(env.data.sensordata, env.sensorranges["velocimeter"])
    s.head_gyro .= view(env.data.sensordata, env.sensorranges["gyro"])
    s.paw_contacts[1] = view(env.data.sensordata, env.sensorranges["palm_L"])[1]
    s.paw_contacts[2] = view(env.data.sensordata, env.sensorranges["palm_R"])[1]
    s.paw_contacts[3] = view(env.data.sensordata, env.sensorranges["sole_L"])[1]
    s.paw_contacts[4] = view(env.data.sensordata, env.sensorranges["sole_R"])[1]
    s.torso_linvel .= view(env.data.sensordata, env.sensorranges["torso"])
    s.torso_xmat .= body_xmat(env, "torso")
    s.torso_height .= subtree_com(env, "torso")[3] .* 10.0
    s.com_target_array .= reshape(future_targets(env, params), :)
    future_quats!(s.xquat_target_array, env, params)
    s.joint_target_array .= reshape(future_joint_pos(env, params), :)
    s.appendages_target_array .= reshape(future_appendages_pos(env, params), :)
    nothing
end

function reward(env::RodentFollowEnv, params)
    target_vec = target_vector(env, params)
    com_reward = exp(-(norm(target_vec)^2) / (params.reward.falloff.com^2))
    angle_reward = exp(-(angle_to_target(env, params)^2) / (params.reward.falloff.rotation^2))
    joint_reward = exp(-joint_error(env, params) / (params.reward.falloff.joint^2))
    append_reward = appendages_reward_static(env, params)
    ctrl_reward = -params.reward.control_cost * norm(env.data.ctrl)^2
    total_reward  = com_reward + angle_reward + joint_reward + append_reward
    total_reward += ctrl_reward + params.reward.alive_bonus
    total_reward #return clamp(total_reward, params.min_reward, Inf)
end

function status(env::RodentFollowEnv, params)
    target_distance = norm(target_vector(env, params))
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

function info!(ct::ComponentTensor, env::RodentFollowEnv, params)
    ct.torso_x .= torso_x(env)
    ct.torso_y .= torso_y(env)
    ct.torso_z .= torso_z(env)
    ct.lifetime .= float(env.lifetime)
    ct.cumulative_reward .= env.cumulative_reward
    ct.actuator_force_sum_sqr .= norm(env.data.actuator_force)^2
    ct.angle_to_target .= angle_to_target(env, params) |> rad2deg
    ct.joint_reward .= exp(-joint_error(env, params) / (params.reward.falloff.joint^2))
    ct.appendages_reward .= appendages_reward_static(env, params)

    cti = com_target_info(env, params)
    ct.target_frame .= cti.target_frame
    ct.target_vec_x .= cti.target_vec_x
    ct.target_vec_y .= cti.target_vec_y
    ct.target_vec_z .= cti.target_vec_z
    ct.target_distance .= cti.target_distance
    ct.distance_from_spawpoint .= cti.distance_from_spawpoint
    ct.joint_error .= cti.joint_error
    ct.lower_arm_R_error .= cti.lower_arm_R_error
    ct.lower_arm_L_error .= cti.lower_arm_L_error
    ct.foot_R_error .= cti.foot_R_error
    ct.foot_L_error .= cti.foot_L_error
    ct.jaw_error .= cti.jaw_error
    nothing
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
torso_x(env::RodentFollowEnv) = subtree_com(env, "torso")[1]
torso_y(env::RodentFollowEnv) = subtree_com(env, "torso")[2]
torso_z(env::RodentFollowEnv) = subtree_com(env, "torso")[3]

#Targets
target_frame(env::RodentImitationEnv) = env.target_frame

function future_targets(env::RodentImitationEnv, params)
    #torso.com
    
    current_com = SVector{3}(subtree_com(env, "torso"))#(torso.com)
    
    current_xmat = SMatrix{3, 3}(reshape(body_xmat(env, "torso"), 3, 3))#(reshape(torso.xmat, 3, 3))

    in_allo_space = SMatrix{3, 5}(view(env.target.com, :, imitation_horizon(env, params), env.target_clip))
    return current_xmat * (in_allo_space .- current_com)
end

function target_vector(env::RodentImitationEnv, params)::SVector{3, Float64}
    target = SVector{3}(view(env.target.com, :, target_frame(env), env.target_clip))
    current = SVector{3}(subtree_com(env, "torso"))
    return target - current
end

function com_target_info(env::RodentFollowEnv, params)
    target_vec = target_vector(env, params)
    dist = norm(target_vec)
    app_error = appendages_error_static(env, params)
    spawn_point = SVector{3}(view(env.target.com, :, 1, env.target_clip))
    com = SVector{3}(subtree_com(env, "torso"))
    (target_frame=target_frame(env),
     target_vec_x=target_vec[1],
     target_vec_y=target_vec[2],
     target_vec_z=target_vec[3],
     target_distance=dist,
     distance_from_spawpoint=norm(spawn_point - com),
     joint_error=sqrt(joint_error(env, params)),
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

function future_quats!(fq, env::RodentImitationEnv, params)
    fqr = reshape(fq, 3, params.imitation.horizon)
    for (i, t) in enumerate(imitation_horizon(env, params))
        MuJoCo.mju_subQuat(view(fqr, :, i),
                           view(env.target.qpos, 4:7, t, env.target_clip),
                           view(env.data.qpos, 4:7))
    end
end

function future_quats(env::RodentImitationEnv, params)
    rot_vec = zeros(3, params.imitation.horizon)
    future_quats!(rot_vec, env, params)
    rot_vec
end

function angle_to_target(env::RodentImitationEnv, params)::Float64
    target_quat = SVector{4}(view(env.target.qpos, 4:7, target_frame(env), env.target_clip))
    current_quat = SVector{4}(body_xquat(env, "torso"))
    quat_prod = target_quat' * current_quat
    return 2*acos(abs(clamp(quat_prod, -1, 1))) #Concerning that clamp is needed here
end

function future_joint_pos(env::RodentImitationEnv, params)
    joint_indices = 8:size(env.target.qpos, 1)
    view(env.target.qpos, joint_indices, imitation_horizon(env, params), env.target_clip)# .- env.data.qpos[joint_indices]
end

function joint_error(env::RodentImitationEnv, params)
    joint_indices = 8:size(env.target.qpos, 1)
    target_joint = view(env.target.qpos, :, target_frame(env), env.target_clip)
    err = 0.0
    for ji in joint_indices
        err += (target_joint[ji] - env.data.qpos[ji])^2
    end
    return err
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


function appendages_pos_static(env::RodentImitationEnv, params)::SMatrix{3, 5, Float64, 15}
    torso_xpos = SVector{3}(body_xpos(env, "torso"))
    torso_xmat = SMatrix{3, 3}(reshape(body_xmat(env, "torso"), 3, 3))
    #apps = map(appendages()) do app_name
    #    app_body_xpos = SVector{3}(body_xpos(env, app_name))
    #    #torso_xmat*(app_body_xpos .- torso_xpos)
    #end
    hcat((torso_xmat*(SVector{3}(body_xpos(env, app)) .- torso_xpos) for app=appendages())...)
    #hcat(apps...)
end

function appendages_error_static(env::RodentImitationEnv, params)
    target = SMatrix{3, 5}(view(env.target.appendages, :, :, target_frame(env), env.target_clip))
    target .- appendages_pos_static(env, params)
end

function appendages_reward_static(env::RodentImitationEnv, params)
    app_error = appendages_error_static(env, params)
    sum(exp(-sum(view(app_error, :, i).^2) / (params.reward.falloff.appendages^2)) for i=1:5) / 5.0
end

function subtree_com(env::RodentImitationEnv, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(env.data, MuJoCo.mjOBJ_BODY, body)+1
    view(env.data.subtree_com, ind, :)
end

function body_xmat(env::RodentImitationEnv, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(env.data, MuJoCo.mjOBJ_BODY, body)+1
    view(env.data.xmat, ind, :)
end

function body_xpos(env::RodentImitationEnv, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(env.data, MuJoCo.mjOBJ_BODY, body)+1
    view(env.data.xpos, ind, :)
end

function body_xquat(env::RodentImitationEnv, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(env.data, MuJoCo.mjOBJ_BODY, body)+1
    view(env.data.xquat, ind, :)
end