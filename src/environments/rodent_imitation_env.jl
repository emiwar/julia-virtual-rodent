
abstract type RodentFollowEnv <: MuJoCoEnv end

mutable struct RodentImitationEnv{ImLen, ImTarget} <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    target::ImTarget
    target_frame::Int64
    target_clip::Int64
    lifetime::Int64
    cumulative_reward::Float64
    sensorranges::Dict{String, UnitRange{Int64}}
end

function RodentImitationEnv(params; target_data="src/environments/assets/diego_curated_snippets.h5")
    if params.physics.body_scale != 1.0
        @warn "Code for rescaling imitation target removed from preprocessing, will use imitation target for scale=1.0." params.physics.body_scale
    end
    model = dm_control_rodent(torque_actuators = params.physics.torque_control,
                              foot_mods = params.physics.foot_mods,
                              scale = params.physics.body_scale,
                              hip_mods = params.physics.hip_mods,
                              physics_timestep = params.physics.timestep,
                              control_timestep = params.physics.timestep * params.physics.n_physics_steps)
    data = MuJoCo.init_data(model)
    target = load_imitation_target(target_data)
    sensorranges = prepare_sensorranges(model, "walker/" .* ["accelerometer", "velocimeter",
                                                             "gyro", "palm_L", "palm_R",
                                                             "sole_L", "sole_R", "torso"])
    env = RodentImitationEnv{params.imitation.horizon, typeof(target)}(model, data, target, 0, 1, 0, 0.0, sensorranges)
    reset!(env, params, rand(1:(size(env.target)[3])), 1)
    return env
end

function clone(env::RodentImitationEnv{ImLen}, params) where ImLen
    new_env = RodentImitationEnv{ImLen, typeof(env.target)}(
        env.model,
        MuJoCo.init_data(env.model),
        env.target,
        1,1,0,0.0,env.sensorranges
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
            joint_vels = reshape(joint_vels_horizon(env), :),
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
    #joint_reward = alt_joint_reward(env, params)
    joint_vel_reward = exp(-joint_vel_error(env) / (params.reward.falloff.joint_vel^2))
    #joint_vel_reward = alt_joint_vel_reward(env, params)
    append_reward = appendages_reward(env, params)
    ctrl_reward = -params.reward.control_cost * norm(env.data.ctrl)^2
    total_reward  = com_reward + angle_reward + joint_reward + joint_vel_reward + append_reward
    total_reward += ctrl_reward + params.reward.alive_bonus
    total_reward #return clamp(total_reward, params.min_reward, Inf)
end

function status(env::RodentFollowEnv, params)
    target_distance = norm(com_error(env))
    if torso_z(env) < params.physics.min_torso_z 
        return TERMINATED
    elseif target_frame(env) + params.imitation.horizon >= size(env.target)[2]
        return TRUNCATED
    elseif target_distance > params.imitation.max_target_distance
        return TERMINATED
    else
        return RUNNING
    end
end

function info(env::RodentFollowEnv, params)
    (
        qpos_root=(@view env.data.qpos[1:7]),
        torso_x=torso_x(env),
        torso_y=torso_y(env),
        torso_z=torso_z(env),
        lifetime=float(env.lifetime),
        cumulative_reward = env.cumulative_reward,
        actuator_force_sum_sqr = norm(env.data.actuator_force)^2,
        angle_to_target = angle_to_target(env) |> rad2deg,
        joint_reward = exp(-joint_error(env) / (params.reward.falloff.joint^2)),
        joint_vel_reward = exp(-joint_vel_error(env) / (params.reward.falloff.joint_vel^2)),
        appendages_reward = appendages_reward(env, params),
        alt_joint_reward = alt_joint_reward(env, params),
        alt_joint_vel_reward = alt_joint_vel_reward(env, params),
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

function reset!(env::RodentFollowEnv, params, next_clip, next_frame)
    env.lifetime = 0
    env.cumulative_reward = 0.0
    env.target_clip = next_clip
    env.target_frame = next_frame

    MuJoCo.reset!(env.model, env.data)
    env.data.qpos .= view(env.target, :qpos, target_frame(env), env.target_clip)
    env.data.qpos[3] += params.physics.spawn_z_offset
    env.data.qvel .= view(env.target, :qvel, target_frame(env), env.target_clip)
    #Run model forward to get correct initial state
    MuJoCo.forward!(env.model, env.data) 
end

function reset!(env::RodentFollowEnv, params)
    if params.imitation.restart_on_reset
        reset!(env, params, rand(1:(size(env.target)[3])), 1)
    else
        reset!(env, params, env.target_clip, env.target_frame)
    end
end

function preprocess_actions(::Type{EnvType}, actions, state , params) where EnvType <: RodentFollowEnv
    return actions
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
target_com(env::RodentImitationEnv, t) = SVector{3}(view(env.target, :com, t, env.target_clip))
function relative_com(env::RodentImitationEnv, t::Int64)
    body_xmat(env, "walker/torso") * (target_com(env, t) - subtree_com(env, "walker/torso"))
end
@generated function com_horizon(env::RodentImitationEnv{N}) where N
    return :(hcat($((:(relative_com(env, target_frame(env) + $i)) for i=1:N)...)))
end
com_error(env::RodentImitationEnv) = target_com(env) - subtree_com(env, "walker/torso")

#Root quaternion target
target_root_quat(env::RodentImitationEnv) = target_root_quat(env, target_frame(env))
target_root_quat(env::RodentImitationEnv, t) = SVector{4}(view(env.target.qpos, :root_quat, t, env.target_clip))
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
    @view env.target.qpos[:joints, t, env.target_clip]
end
function joints_horizon(env::RodentImitationEnv)
    @view env.target.qpos[:joints, imitation_horizon(env), env.target_clip]
end
function joint_error(env::RodentImitationEnv)
    joint_indices = 8:size(env.target.qpos)[1]
    target_joint = view(env.target, :qpos, target_frame(env), env.target_clip)
    err = 0.0
    for ji in joint_indices
        err += (target_joint[ji] - env.data.qpos[ji])^2
    end
    return err
end
function alt_joint_reward(env::RodentImitationEnv, params)
    joint_indices = 8:size(env.target.qpos)[1]
    target_joint = view(env.target, :qpos, target_frame(env), env.target_clip)
    rew = 0.0
    sig_sqr = params.reward.falloff.per_joint^2
    for ji in joint_indices
        rew += exp(-(target_joint[ji] - env.data.qpos[ji])^2 / sig_sqr)
    end
    return rew / length(joint_indices)
end

#Joint vel
function target_joint_vels(env::RodentImitationEnv, t)
    @view env.target.qvel[:joints, t, env.target_clip]
end
function joint_vels_horizon(env::RodentImitationEnv)
    @view env.target.qvel[:joints_vel, imitation_horizon(env), env.target_clip]
end
function joint_vel_error(env::RodentImitationEnv)
    joint_indices = 7:size(env.target.qvel)[1]
    target_joint_vel = view(env.target, :qvel, target_frame(env), env.target_clip)
    err = 0.0
    for ji in joint_indices
        err += (target_joint_vel[ji] - env.data.qvel[ji])^2
    end
    return err
end
function alt_joint_vel_reward(env::RodentImitationEnv, params)
    joint_indices = 7:size(env.target.qvel)[1]
    target_joint_vel = view(env.target, :qvel, target_frame(env), env.target_clip)
    rew = 0.0
    sig_sqr = params.reward.falloff.per_joint_vel^2
    for ji in joint_indices
        rew += exp(-(target_joint_vel[ji] - env.data.qvel[ji])^2 / sig_sqr)
    end
    return rew / length(joint_indices)
end

#Appendages
function appendage_pos(env::RodentImitationEnv, appendage_name::String)
    body_xmat(env, "walker/torso")*(body_xpos(env, appendage_name) .- body_xpos(env, "walker/torso"))
end
@generated function appendages_pos(env::RodentImitationEnv)
    :(hcat($((:(appendage_pos(env, "walker/" * $a)) for a=appendages_order())...)))
end
appendages_pos_horizon(env) = appendages_pos_horizon(env, target_frame(env))
function appendages_pos_horizon(env, t)
    SMatrix{3, length(appendages_order())}(view(env.target, :appendages, t, env.target_clip))
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
    return reward / length(appendages_order())
end

function com_target_info(env::RodentFollowEnv, params)
    target_vec = target_com(env) - subtree_com(env, "walker/torso")
    dist = norm(target_vec)
    app_error = appendages_error(env)
    spawn_point = target_com(env, 1)
    (target_frame=float(target_frame(env)),
     target_vec_x=target_vec[1],
     target_vec_y=target_vec[2],
     target_vec_z=target_vec[3],
     target_distance=dist,
     distance_from_spawpoint=norm(spawn_point - subtree_com(env, "walker/torso")),
     joint_error=sqrt(joint_error(env)),
     joint_vel_error=sqrt(joint_vel_error(env)),
     lower_arm_R_error=norm(view(app_error, :, 1)),
     lower_arm_L_error=norm(view(app_error, :, 2)),
     foot_R_error=norm(view(app_error, :, 3)),
     foot_L_error=norm(view(app_error, :, 4)),
     jaw_error=norm(view(app_error, :, 5)),
     all_bodies_error(env)...
     )
end

function all_bodies_error(env::RodentImitationEnv)
    map(bodies_order()) do body_name
        target_pos = SVector{3}(view(env.target.body_positions, Symbol(body_name),
                                     target_frame(env), env.target_clip))
        current_pos = body_xpos(env, "walker/"*body_name)
        error = norm(current_pos - target_pos)
        Symbol("global_error_" * body_name) => error
    end
end

function Base.show(io::IO, env::RodentImitationEnv{ImLen}) where ImLen
    compact = get(io, :compact, false)
    if compact
        print(io, "RodentImitationEnv{ImLen = $ImLen}")
    else
        println(io, "RodentImitationEnv{ImLen = $ImLen}:")
        println(io, "\ttarget_clip: $(env.target_clip)")
        println(io, "\ttarget_frame: $(env.target_frame)")
        println(io, "\tlifetime: $(env.lifetime)")
        println(io, "\tcumulative_reward: $(env.cumulative_reward)")
    end
end
