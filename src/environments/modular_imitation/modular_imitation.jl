@concrete struct ModularImitationEnv <: AbstractEnv
    walker<:Walker
    target_poses<:ComponentArray
    target_sensors<:ComponentArray
    max_target_distance::Float64
    restart_on_reset::Bool
    target_fps::Float64
    target_timepoint::Base.RefValue{Float64}
    target_clip::Base.RefValue{Int64}
    lifetime::Base.RefValue{Int64}
end

function ModularImitationEnv(walker; max_target_distance::Float64,
                             restart_on_reset::Bool=true, target_fps::Float64=50.0)
    target_poses = load_imitation_target(walker)[(:qpos, :qvel), :, :]
    target_sensors = precompute_target(walker, target_poses)
    target_timepoint = Ref(0.0)
    target_clip = Ref(0)
    lifetime = Ref(0)
    env = ModularImitationEnv(walker, target_poses, target_sensors, max_target_distance,
                              restart_on_reset, target_fps, target_timepoint, target_clip, lifetime)
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
                #elbow_pos = target.arm_L.egocentric_elbow_pos,
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
                #elbow_pos = target.arm_R.egocentric_elbow_pos,
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
                pelvis_z = target.leg_L.pelvis_zaxis,
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
                pelvis_z = target.leg_R.pelvis_zaxis,
            )
        ),
        torso = (
            proprioception = prop.torso,
            imitation_target = (
                zaxis = target.torso.zaxis,
                lumbar_bend   = target.torso.lumbar_bend,
                lumbar_twist  = target.torso.lumbar_twist,
                lumbar_extend = target.torso.lumbar_extend,
                height_above_ground = target.torso.height_above_ground,
            )
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
    reward_shape(a, b) = exp(-(norm(a-b)/0.5))
    cosine_dist(a, b) = dot(a, b) / norm(a) / norm(b)
    prop = proprioception(env.walker)
    target = get_current_target(env)
    (
        hand_L = (
            finger_joint = reward_shape(prop.hand_L.finger_angle, target.hand_L.finger_angle),
            wrist_joint = reward_shape(prop.hand_L.wrist_angle, target.hand_L.wrist_angle),
            orientation = dot(prop.hand_L.xaxis, target.hand_L.xaxis),
            #hand_pos = reward_shape(prop.hand_L.egocentric_hand_pos, target.hand_L.egocentric_hand_pos),
        ),
        arm_L = (
            hand_pos = reward_shape(prop.arm_L.egocentric_hand_pos, target.arm_L.egocentric_hand_pos),
            #elbow_pos = reward_shape(prop.arm_L.egocentric_elbow_pos, target.arm_L.egocentric_elbow_pos),
            elbow_joint = reward_shape(prop.hand_L.elbow_angle, target.hand_L.elbow_angle),
        ),
        hand_R = (
            finger_joint = reward_shape(prop.hand_R.finger_angle, target.hand_R.finger_angle),
            wrist_joint = reward_shape(prop.hand_R.wrist_angle, target.hand_R.wrist_angle),
            orientation = dot(prop.hand_R.xaxis, target.hand_R.xaxis),
            #hand_pos = reward_shape(prop.hand_R.egocentric_hand_pos, target.hand_R.egocentric_hand_pos),
        ),
        arm_R = (
            hand_pos = reward_shape(prop.arm_R.egocentric_hand_pos, target.arm_R.egocentric_hand_pos),
            #elbow_pos = reward_shape(prop.arm_R.egocentric_elbow_pos, target.arm_R.egocentric_elbow_pos),
            elbow_joint = reward_shape(prop.hand_R.elbow_angle, target.hand_R.elbow_angle),
        ),
        foot_L = (
            toe_joint = reward_shape(prop.foot_L.toe_angle, target.foot_L.toe_angle),
            ankle_joint = reward_shape(prop.foot_L.ankle_angle, target.foot_L.ankle_angle),
            orientation = dot(prop.foot_L.xaxis, target.foot_L.xaxis),
            #foot_pos = reward_shape(prop.foot_L.egocentric_foot_pos, target.foot_L.egocentric_foot_pos),
        ),
        leg_L = (
            knee_joint = reward_shape(prop.leg_L.knee_angle, target.leg_L.knee_angle),
            #knee_pos = reward_shape(prop.leg_L.egocentric_knee_pos, target.leg_L.egocentric_knee_pos),
            foot_pos = reward_shape(prop.leg_L.egocentric_foot_pos, target.leg_L.egocentric_foot_pos),
            #orientation = dot(prop.foot_L.xaxis, target.foot_L.xaxis),
            pelvis_z = dot(prop.leg_L.pelvis_zaxis, target.leg_L.pelvis_zaxis),
        ),
        foot_R = (
            toe_joint = reward_shape(prop.foot_R.toe_angle, target.foot_R.toe_angle),
            ankle_joint = reward_shape(prop.foot_R.ankle_angle, target.foot_R.ankle_angle),
            orientation = dot(prop.foot_R.xaxis, target.foot_R.xaxis),
            #foot_pos = reward_shape(prop.foot_R.egocentric_foot_pos, target.foot_R.egocentric_foot_pos),
        ),
        leg_R = (
            knee_joint = reward_shape(prop.leg_R.knee_angle, target.leg_R.knee_angle),
            #knee_pos = reward_shape(prop.leg_R.egocentric_knee_pos, target.leg_R.egocentric_knee_pos),
            foot_pos = reward_shape(prop.leg_R.egocentric_foot_pos, target.leg_R.egocentric_foot_pos),
            #orientation = dot(prop.foot_R.xaxis, target.foot_R.xaxis),
            pelvis_z = dot(prop.leg_R.pelvis_zaxis, target.leg_R.pelvis_zaxis),
        ),
        torso = (
            orientation_z = cosine_dist(prop.torso.zaxis, target.torso.zaxis),
            height_above_ground = reward_shape(prop.torso.height_above_ground, target.torso.height_above_ground),
            lumbar_bend   = reward_shape(prop.torso.lumbar_bend,   target.torso.lumbar_bend),
            lumbar_twist  = reward_shape(prop.torso.lumbar_twist,  target.torso.lumbar_twist),
            lumbar_extend = reward_shape(prop.torso.lumbar_extend, target.torso.lumbar_extend),
        ),
        head = (
            orientation_x = cosine_dist(prop.head.xaxis, target.head.xaxis),
            orientation_z = cosine_dist(prop.head.zaxis, target.head.zaxis),
            head_pos = reward_shape(prop.head.egocentric_pos, target.head.egocentric_pos),
            mandible = reward_shape(prop.head.mandible, target.head.mandible),
        )
    )
end

reward(env::ModularImitationEnv) = map(sum, compute_rewards(env))

function status(env::ModularImitationEnv)
    prop = proprioception(env.walker)
    target = get_current_target(env)
    if dot(prop.torso.zaxis, target.torso.zaxis) < 0.5
        return TERMINATED
    elseif dot(prop.leg_L.pelvis_zaxis, target.leg_L.pelvis_zaxis) < 0.5
        return TERMINATED
    elseif prop.torso.height_above_ground < env.walker.min_torso_z
        return TERMINATED
    elseif target_frame(env)+1 >= clip_length(env)
        return TRUNCATED
    #elseif target_distance > env.max_target_distance
    #    return TERMINATED
    else
        return RUNNING
    end
end

function info(env::ModularImitationEnv)
    prop = proprioception(env.walker)
    target = get_current_target(env)
    (
        info(env.walker)...,   
        lifetime = float(env.lifetime[]),
        head_pos_error   = norm(prop.head.egocentric_pos - target.head.egocentric_pos),
        hand_L_pos_error = norm(prop.hand_L.egocentric_hand_pos - target.hand_L.egocentric_hand_pos),
        foot_L_pos_error = norm(prop.foot_L.egocentric_foot_pos - target.foot_L.egocentric_foot_pos),
        finger_L_joint_err = first(prop.hand_L.finger_angle - target.hand_L.finger_angle),
        wrist_L_joint_err  = first(prop.hand_L.wrist_angle  - target.hand_L.wrist_angle),
        torso_height_above_ground = prop.torso.height_above_ground,
        torso_z_dot = prop.torso.zaxis[3],
        pelvis_z_dot = dot(prop.leg_R.pelvis_zaxis, target.leg_R.pelvis_zaxis),
        #cumulative_reward = env.cumulative_reward,
        reward_terms = compute_rewards(env),
    )
end

function act!(env::ModularImitationEnv, action)
    set_ctrl!(env.walker, action)
    for _=1:env.walker.n_physics_steps
        step!(env.walker)
    end
    env.target_timepoint[] += dt(env.walker) * env.walker.n_physics_steps
    env.lifetime[] += 1
    #env.cumulative_reward .+= reward(env)
end

function reset!(env::ModularImitationEnv, next_clip, next_frame)
    env.lifetime[] = 0
    #env.cumulative_reward .= 0.0
    env.target_clip[] = next_clip
    env.target_timepoint[] = next_frame / env.target_fps

    start_qpos = view(env.target_poses.qpos, :, target_frame(env), target_clip(env))
    start_qvel = view(env.target_poses.qvel, :, target_frame(env), target_clip(env))
    reset!(env.walker, start_qpos, start_qvel)
end

function reset!(env::ModularImitationEnv)
    if env.restart_on_reset
        reset!(env, rand(1:n_clips(env)), 1)
    else
        reset!(env, env.target_clip[], target_frame(env))
    end
end

function duplicate(env::ModularImitationEnv)
    new_env = ModularImitationEnv(
        clone(env.walker), env.target_poses, env.target_sensors,
        env.max_target_distance, env.restart_on_reset, env.target_fps,
        Ref(0.0), Ref(0), Ref(0)#, zero(env.cumulative_reward)
    )
    reset!(new_env)
    return new_env
end

null_action(env::ModularImitationEnv) = null_action(env.walker)

clip_length(env::ModularImitationEnv) = size(env.target_sensors)[2]
n_clips(env::ModularImitationEnv) = size(env.target_sensors)[3]
target_frame(env::ModularImitationEnv) = round(Int64, env.target_timepoint[] * env.target_fps)
target_clip(env::ModularImitationEnv) = env.target_clip[]

function get_current_target(env::ModularImitationEnv)
    precise_frame = env.target_timepoint[] * env.target_fps
    int_frame = floor(Int64, precise_frame)
    target = view(env.target_sensors, :, int_frame, env.target_clip[])
    next_target = view(env.target_sensors, :, int_frame+1, env.target_clip[])
    frac_frame = precise_frame - int_frame
    interpolated = (1-frac_frame) .* target .+ frac_frame .* next_target
    return interpolated
end

function precompute_target(walker::ModularRodent, orig_target::ComponentArray)
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
    #ImLen = im_len(env)
    compact = get(io, :compact, false)
    if compact
        print(io, "ModularImitationEnv{Walker=$(env.walker)}")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)ModularImitationEnv")
        indented_io = IOContext(io, :indent => (get(io, :indent, 0) + 2))
        show(indented_io, env.walker)
        println(io, "$(indent)  lifetime: $(env.lifetime[])")
        #println(io, "$(indent)  cumulative_reward: $(env.cumulative_reward)")
    end
end
