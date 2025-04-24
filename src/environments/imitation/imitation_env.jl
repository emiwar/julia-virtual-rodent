struct ImitationEnv{W, IRS, ImLen, ImTarget} <: AbstractEnv
    walker::W
    reward_spec::IRS
    target::ImTarget
    max_target_distance::Float64
    restart_on_reset::Bool
    target_frame::Ref{Int64}
    target_clip::Ref{Int64}
    lifetime::Ref{Int64}
    cumulative_reward::Ref{Float64}
end

function ImitationEnv(walker, reward_spec, target; horizon::Int64,
                      max_target_distance::Float64, restart_on_reset::Bool=true)
    target_frame = Ref(0)
    target_clip = Ref(0)
    lifetime = Ref(0)
    cumulative_reward = Ref(0.0)
    env = ImitationEnv{typeof(walker), typeof(reward_spec), horizon, typeof(target)}(
            walker, reward_spec, target, max_target_distance, restart_on_reset,
            target_frame, target_clip, lifetime, cumulative_reward)
    reset!(env)
    return env
end

function state(env::ImitationEnv)
    (
        proprioception = proprioception(env.walker),
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
function reward(env::ImitationEnv)
    return sum(compute_rewards(env))
end

function compute_rewards(env::ImitationEnv)
    #Specialized in `imitation_reward_spec.jl`
    error("`compute_rewards` must be specialized for each walker and reward spec.")
end

function status(env::ImitationEnv)
    target_distance = norm(com_error(env))
    if torso_z(env.walker) < min_torso_z(env.walker)
        return TERMINATED
    elseif imitation_horizon(env)[end] >= clip_length(env)
        return TRUNCATED
    elseif target_distance > env.max_target_distance
        return TERMINATED
    else
        return RUNNING
    end
end

function info(env::ImitationEnv)
    (
        info(env.walker)...,   
        lifetime = float(env.lifetime[]),
        cumulative_reward = env.cumulative_reward[],
        compute_rewards(env)...,
        target_info(env)...
    )
end

#Actions
function act!(env::ImitationEnv, action)
    set_ctrl!(env.walker, action)
    for _=1:env.walker.n_physics_steps
        step!(env.walker)
    end
    env.lifetime[] += 1

    #TODO: Make this more correct and elegant
    if env.lifetime[] % 2 == 0 #Hack since target update each 20ms but simulation each 10ms (but this depends on params so shouldn't be hard-coded here)
        env.target_frame[] += 1
    end

    env.cumulative_reward[] += reward(env)
end

function reset!(env::ImitationEnv, next_clip, next_frame)
    env.lifetime[] = 0
    env.cumulative_reward[] = 0.0
    env.target_clip[] = next_clip
    env.target_frame[] = next_frame

    start_qpos = view(env.target, :qpos, target_frame(env), target_clip(env))
    start_qvel = view(env.target, :qvel, target_frame(env), target_clip(env))
    reset!(env.walker, start_qpos, start_qvel)
end

function reset!(env::ImitationEnv)
    if env.restart_on_reset
        reset!(env, rand(1:n_clips(env)), 1)
    else
        reset!(env, env.target_clip[], target_frame(env))
    end
end


function duplicate(env::ImitationEnv{W, IRS, ImLen, ImTarget}) where {W, IRS, ImLen, ImTarget}
    new_env = ImitationEnv{W, IRS, ImLen, ImTarget}(
        clone(env.walker), env.reward_spec, env.target,
        env.max_target_distance, env.restart_on_reset,
        Ref(0), Ref(0), Ref(0), Ref(0.0)
    )
    reset!(new_env)
    return new_env
end

null_action(env::ImitationEnv) = null_action(env.walker)

#Targets
clip_length(env::ImitationEnv) = size(env.target)[2]
n_clips(env::ImitationEnv) = size(env.target)[3]
target_frame(env::ImitationEnv) = env.target_frame[]
target_clip(env::ImitationEnv) = env.target_clip[]
im_len(env::ImitationEnv{W, IRS, ImLen, ImTarget}) where {W, IRS, ImLen, ImTarget} = ImLen

function imitation_horizon(env::ImitationEnv)
    target_frame(env) .+ (1:im_len(env))
end

#Center-of-mass target
target_com(env::ImitationEnv) = target_com(env, target_frame(env))
target_com(env::ImitationEnv, t) = SVector{3}(view(env.target, :com, t, target_clip(env)))
function relative_com(env::ImitationEnv, t::Int64)
    body_xmat(env.walker, "walker/torso") * (target_com(env, t) - subtree_com(env.walker, "walker/torso"))
end
@generated function com_horizon(env::ImitationEnv{W, IRS, ImLen, ImTarget}) where {W, IRS, ImLen, ImTarget}
    return :(hcat($((:(relative_com(env, target_frame(env) + $i)) for i=1:ImLen)...)))
end
com_error(env::ImitationEnv) = target_com(env) - subtree_com(env.walker, "walker/torso")

#Root quaternion target
target_root_quat(env::ImitationEnv) = target_root_quat(env, target_frame(env))
target_root_quat(env::ImitationEnv, t) = SVector{4}(view(env.target.qpos, :root_quat, t, target_clip(env)))
function relative_root_quat(env::ImitationEnv, t::Int64)::SVector{3, Float64}
    subQuat(target_root_quat(env, t), body_xquat(env.walker, "walker/torso"))
end
@generated function root_quat_horizon(env::ImitationEnv{W, IRS, ImLen, ImTarget}) where {W, IRS, ImLen, ImTarget}
    return :(hcat($((:(relative_root_quat(env, target_frame(env) + $i)) for i=1:ImLen)...)))
end
function angle_to_target(env::ImitationEnv)
    quat_prod = target_root_quat(env)' * body_xquat(env.walker, "walker/torso")
    return 2*acos(abs(clamp(quat_prod, -1, 1))) #Concerning that clamp is needed here
end

#Joints
function target_joints(env::ImitationEnv, t)
    @view env.target.qpos[:joints, t, target_clip(env)]
end
function joints_horizon(env::ImitationEnv)
    @view env.target.qpos[:joints, imitation_horizon(env), target_clip(env)]
end
function joint_error(env::ImitationEnv)
    target_joint = view(env.target, :qpos, target_frame(env), target_clip(env))
    err = 0.0
    for ji in joint_indices(env.walker)
        err += (target_joint[ji] - env.walker.data.qpos[ji])^2
    end
    return err
end

#Joint vel
function target_joint_vels(env::ImitationEnv, t)
    @view env.target.qvel[:joints, t, env.target_clip]
end
function joint_vels_horizon(env::ImitationEnv)
    @view env.target.qvel[:joints_vel, imitation_horizon(env), target_clip(env)]
end
function joint_vel_error(env::ImitationEnv)
    target_joint_vel = view(env.target, :qvel, target_frame(env), target_clip(env))
    err = 0.0
    for ji in joint_vel_indices(env.walker)
        err += (target_joint_vel[ji] - env.walker.data.qvel[ji])^2
    end
    return err
end

appendages_pos_horizon(env::ImitationEnv) = appendages_pos_horizon(env, target_frame(env))
function appendages_pos_horizon(env::ImitationEnv, t)
    n_appendages = length(appendages_order(env.walker))
    SMatrix{3, n_appendages}(view(env.target, :appendages, t, target_clip(env)))
end
function appendages_error(env::ImitationEnv)
    appendages_pos_horizon(env) - appendages_pos(env.walker)
end

function target_info(env::ImitationEnv)
    target_vec = target_com(env) - subtree_com(env.walker, "walker/torso")
    dist = norm(target_vec)
    app_error = appendages_error(env)
    spawn_point = target_com(env, 1)
    (target_frame=float(target_frame(env)),
     target_vec_x=target_vec[1],
     target_vec_y=target_vec[2],
     target_vec_z=target_vec[3],
     target_distance=dist,
     angle_to_target = angle_to_target(env) |> rad2deg,
     distance_from_spawpoint=norm(spawn_point - subtree_com(env.walker, "walker/torso")),
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

function all_bodies_error(env::ImitationEnv)
    map(bodies_order(env.walker)) do body_name
        target_pos = SVector{3}(view(env.target.body_positions, Symbol(body_name),
                                     target_frame(env), target_clip(env)))
        current_pos = body_xpos(env.walker, "walker/"*body_name)
        error = norm(current_pos - target_pos)
        Symbol("global_error_" * body_name) => error
    end
end

function Base.show(io::IO, env::ImitationEnv)
    ImLen = im_len(env)
    compact = get(io, :compact, false)
    if compact
        print(io, "ImitationEnv{Walker=")
        print(io, env.walker)
        print(io, ", ImLen = $ImLen}")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)ImitationEnv{ImLen=$ImLen}")
        indented_io = IOContext(io, :indent => (get(io, :indent, 0) + 2))
        show(indented_io, env.walker)
        show(indented_io, env.reward_spec)
        println(io, "$(indent)  target_clip: $(env.target_clip[])")
        println(io, "$(indent)  target_frame: $(env.target_frame[])")
        println(io, "$(indent)  lifetime: $(env.lifetime[])")
        println(io, "$(indent)  cumulative_reward: $(env.cumulative_reward[])")
    end
end
