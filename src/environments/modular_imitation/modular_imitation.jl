struct ModularImitationEnv{W, ImTarget, CR} <: AbstractEnv
    walker::W
    target::ImTarget
    max_target_distance::Float64
    restart_on_reset::Bool
    target_fps::Float64
    target_timepoint::Base.RefValue{Float64}
    target_clip::Base.RefValue{Int64}
    lifetime::Base.RefValue{Int64}
    cumulative_reward::CR
end

function ModularImitationEnv(walker; max_target_distance::Float64,
                             restart_on_reset::Bool=true, target_fps::Float64=50.0)
    target = precompute_target(walker)
    target_timepoint = Ref(0.0)
    target_clip = Ref(0)
    lifetime = Ref(0)
    cumulative_reward = nothing
    env = ModularImitationEnv{typeof(walker), typeof(target), typeof(cumulative_reward)}(
            walker, target, max_target_distance, restart_on_reset, target_fps,
            target_timepoint, target_clip, lifetime, cumulative_reward)
    reset!(env)
    return env
end

function state(env::ModularImitationEnv)
    prop = proprioception(env.walker)
    target = get_current_target(env)
    (
        hand_L = (
            proprioception = prop.hand_L,
            imitation_target = (
                pos = target.hand_L.egocentric_hand_pos,
                wrist_angle = target.hand_L.wrist_angle,
                finger_angle = target.hand_L.finger_angle,
                xaxis = target.hand_L.xaxis
            )
        ),
        arm_L = (
            proprioception = prop.arm_L,
            imitation_target = (
                hand_pos = target.arm_L.egocentric_hand_pos,
                elbow_pos = target.arm_L.egocentric_elbow_pos,
                elbow_angle = target.arm_L.elbow_angle,
            )
        ),
        hand_R = (
            proprioception = prop.hand_R,
            imitation_target = (
                pos = target.hand_R.egocentric_hand_pos,
                wrist_angle = target.hand_R.wrist_angle,
                finger_angle = target.hand_R.finger_angle,
                xaxis = target.hand_R.xaxis
            )
        ),
        arm_R = (
            proprioception = prop.arm_R,
            imitation_target = (
                hand_pos = target.arm_R.egocentric_hand_pos,
                elbow_pos = target.arm_R.egocentric_elbow_pos,
                elbow_angle = target.arm_R.elbow_angle,
            )
        ),
        foot_L = (
            proprioception = prop.foot_L,
            imitation_target = (
                pos = target.foot_L.egocentric_foot_pos,
                toe_angle = target.foot_L.toe_angle,
                ankle_angle = target.foot_L.ankle_angle,
                xaxis = target.foot_L.xaxis
            )
        ),
        leg_L = (
            proprioception = prop.leg_L,
            imitation_target = (
                foot_pos = target.leg_L.egocentric_foot_pos,
                knee_angle = target.leg_L.knee_angle,
            )
        ),
        foot_R = (
            proprioception = prop.foot_R,
            imitation_target = (
                pos = target.foot_R.egocentric_foot_pos,
                toe_angle = target.foot_R.toe_angle,
                ankle_angle = target.foot_R.ankle_angle,
                xaxis = target.foot_R.xaxis
            )
        ),
        leg_R = (
            proprioception = prop.leg_R,
            imitation_target = (
                foot_pos = target.leg_R.egocentric_foot_pos,
                knee_angle = target.leg_R.knee_angle,
            )
        ),
        torso = (
            proprioception = prop.torso,
            imitation_target = (accelerometer = target.torso.accelerometer,)
        ),
        head = (
            proprioception = prop.head,
            imitation_target = (
                accelerometer = target.head.accelerometer,
                egocentric_pos = target.head.egocentric_pos,
                mandible = target.head.mandible
            )
        )
    )
end

function compute_rewards(env::ModularImitationEnv)
    reward_shape(a, b) = exp(-(norm(a-b)/0.25)^2)
    cosine_dist(a, b) = dot(a, b) / norm(a) / norm(b)
    prop = proprioception(env.walker)
    target = get_current_target(env)
    (
        hand_L = (
            finger_joint = reward_shape(prop.hand_L.finger_angle, target.hand_L.finger_angle),
            wrist_joint = reward_shape(prop.hand_L.wrist_angle, target.hand_L.wrist_angle),
            orientation = dot(prop.hand_L.xaxis, target.hand_L.xaxis),
            hand_pos = reward_shape(prop.hand_L.egocentric_hand_pos, target.hand_L.egocentric_hand_pos),
        ),
        arm_L = (
            hand_pos = reward_shape(prop.arm_L.egocentric_hand_pos, target.arm_L.egocentric_hand_pos),
            elbow_pos = reward_shape(prop.arm_L.egocentric_elbow_pos, target.arm_L.egocentric_elbow_pos),
            elbow_joint = reward_shape(prop.hand_L.elbow_angle, target.hand_L.elbow_angle),
        ),
        hand_R = (
            finger_joint = reward_shape(prop.hand_R.finger_angle, target.hand_R.finger_angle),
            wrist_joint = reward_shape(prop.hand_R.wrist_angle, target.hand_R.wrist_angle),
            orientation = dot(prop.hand_R.xaxis, target.hand_R.xaxis),
            hand_pos = reward_shape(prop.hand_R.egocentric_hand_pos, target.hand_R.egocentric_hand_pos),
        ),
        arm_R = (
            hand_pos = reward_shape(prop.arm_R.egocentric_hand_pos, target.arm_R.egocentric_hand_pos),
            elbow_pos = reward_shape(prop.arm_R.egocentric_elbow_pos, target.arm_R.egocentric_elbow_pos),
            elbow_joint = reward_shape(prop.hand_R.elbow_angle, target.hand_R.elbow_angle),
        ),
        foot_L = (
            toe_joint = reward_shape(prop.foot_L.toe_angle, target.foot_L.toe_angle),
            ankle_joint = reward_shape(prop.foot_L.ankle_angle, target.foot_L.ankle_angle),
            orientation = dot(prop.foot_L.xaxis, target.foot_L.xaxis),
            foot_pos = reward_shape(prop.foot_L.egocentric_foot_pos, target.foot_L.egocentric_foot_pos),
        ),
        leg_L = (
            knee_joint = reward_shape(prop.leg_L.knee_angle, target.leg_L.knee_angle),
            foot_pos = reward_shape(prop.leg_L.egocentric_hand_pos, target.leg_L.egocentric_hand_pos),
        ),
        foot_R = (
            toe_joint = reward_shape(prop.foot_R.toe_angle, target.foot_R.toe_angle),
            ankle_joint = reward_shape(prop.foot_R.ankle_angle, target.foot_R.ankle_angle),
            orientation = dot(prop.foot_R.xaxis, target.foot_R.xaxis),
            foot_pos = reward_shape(prop.foot_R.egocentric_foot_pos, target.foot_R.egocentric_foot_pos),
        ),
        leg_R = (
            knee_joint = reward_shape(prop.leg_R.knee_angle, target.leg_R.knee_angle),
            foot_pos = reward_shape(prop.leg_R.egocentric_hand_pos, target.leg_R.egocentric_hand_pos),
        ),
        torso = (
            orientation = cosine_dist(prop.torso.accelerometer, target.torso.accelerometer)
        ),
        head = (
            orientation = cosine_dist(prop.head.accelerometer, target.head.accelerometer),
            head_pos = reward_shape(prop.head.egocentric_pos, target.head.egocentric_pos)
        )
    )
end

reward(env::ModularImitationEnv) = map(sum, compute_rewards(env))

function status(env::ModularImitationEnv)
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

function info(env::ModularImitationEnv)
    (
        info(env.walker)...,   
        lifetime = float(env.lifetime[]),
        cumulative_reward = env.cumulative_reward,
        compute_rewards(env)...
    )
end

function act!(env::ModularImitationEnv, action)
    set_ctrl!(env.walker, action)
    for _=1:env.walker.n_physics_steps
        step!(env.walker)
    end
    env.target_timepoint[] += dt(env.walker) * env.walker.n_physics_steps
    env.lifetime[] += 1
    env.cumulative_reward .+= reward(env)
end

function reset!(env::ModularImitationEnv, next_clip, next_frame)
    env.lifetime[] = 0
    env.cumulative_reward .= 0.0
    env.target_clip[] = next_clip
    env.target_timepoint[] = next_frame / env.target_fps

    start_qpos = view(env.target.qpos, :, target_frame(env), target_clip(env))
    start_qvel = view(env.target.qvel, :, target_frame(env), target_clip(env))
    reset!(env.walker, start_qpos, start_qvel)
end

function reset!(env::ModularImitationEnv)
    if env.restart_on_reset
        reset!(env, rand(1:n_clips(env)), 1)
    else
        reset!(env, env.target_clip[], target_frame(env))
    end
end

function duplicate(env::ModularImitationEnv{W, ImTarget, CR}) where {W, ImTarget, CR}
    new_env = ModularImitationEnv{W, ImTarget, CR}(
        clone(env.walker), env.target,
        env.max_target_distance, env.restart_on_reset, env.target_fps,
        Ref(0.0), Ref(0), Ref(0), zero(env.cumulative_reward)
    )
    reset!(new_env)
    return new_env
end

null_action(env::ModularImitationEnv) = null_action(env.walker)

clip_length(env::ModularImitationEnv) = size(env.target)[2]
n_clips(env::ModularImitationEnv) = size(env.target)[3]
target_frame(env::ModularImitationEnv) = round(Int64, env.target_timepoint[] * env.target_fps)
target_clip(env::ModularImitationEnv) = env.target_clip[]

function get_current_target(env::ModularImitationEnv)
    precise_frame = env.target_timepoint[] * env.target_fps
    int_frame = floor(Int64, precise_frame)
    target = view(env.target, :, int_frame, env.target_clip[])
    next_target = view(env.target, :, int_frame+1, env.target_clip[])
    frac_frame = precise_frame - int_frame
    interpolated = (1-frac_frame) .* target .+ frac_frame .* next_target
    return interpolated
end

function precompute_target(walker::ModularRodent)
    orig_target = load_imitation_target(walker)
    _, dur, n_clips = size(orig_target)
    template_prop = proprioception(walker) |> ComponentArray
    new_target = ComponentArray(zeros(length(template_prop), dur, n_clips),
                                getaxes(template_prop)[1], FlatAxis(), FlatAxis())
    @Threads.threads for i = 1:n_clips
        w = clone(walker)
        for t = 1:dur
            w.data.qpos .= @view orig_target.qpos[:, t, i]
            w.data.qvel .= @view orig_target.qvel[:, t, i]
            MuJoCo.forward!(w.model, w.data)
            new_target[:, t, i] = proprioception(w)
        end
    end
    return new_target
end

function Base.show(io::IO, env::ModularImitationEnv)
    ImLen = im_len(env)
    compact = get(io, :compact, false)
    if compact
        print(io, "ModularImitationEnv{Walker=")
        print(io, env.walker)
        print(io, ", ImLen = $ImLen}")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)ModularImitationEnv{ImLen=$ImLen}")hand
        show(indented_io, env.reward_spec)hand
        println(io, "$(indent)  lifetime: $(env.lifetime[])")
        println(io, "$(indent)  cumulative_reward: $(env.cumulative_reward)")
    end
end
