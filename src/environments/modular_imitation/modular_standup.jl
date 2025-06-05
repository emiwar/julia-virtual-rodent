@concrete terse struct ModularStandupEnv <: AbstractEnv
    walker<:Walker
    target<:ComponentVector
    lifetime::Base.RefValue{Int64}
    randomTrunc::Base.RefValue{Bool}
end

function ModularStandupEnv(walker)
    reset!(walker)
    #walker_clone = clone(walker)
    #MuJoCo.forward!(walker_clone.model, walker_clone.data)
    target = ComponentVector(proprioception(walker))
    env = ModularStandupEnv(walker, target, Ref(0), Ref(false))
    reset!(env)
    return env
end

function state(env::ModularStandupEnv)
    prop = proprioception(env.walker)
    target = env.target
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
            imitation_target = (
                zaxis = target.torso.zaxis,
                lumbar_bend   = target.torso.lumbar_bend,
                lumbar_twist  = target.torso.lumbar_twist,
                lumbar_extend = target.torso.lumbar_extend,
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

function compute_rewards(env::ModularStandupEnv)
    reward_shape(a, b) = exp(-(norm(a-b)/0.5))#exp(-(norm(a-b)/0.25)^2)
    cosine_dist(a, b) = dot(a, b) / norm(a) / norm(b)
    prop = proprioception(env.walker)
    target = env.target
    (
        hand_L = (
            finger_joint = reward_shape(prop.hand_L.finger_angle, target.hand_L.finger_angle),
            wrist_joint = reward_shape(prop.hand_L.wrist_angle, target.hand_L.wrist_angle),
            orientation = dot(prop.hand_L.xaxis, target.hand_L.xaxis),
            #hand_pos = reward_shape(prop.hand_L.egocentric_hand_pos, target.hand_L.egocentric_hand_pos),
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
            #hand_pos = reward_shape(prop.hand_R.egocentric_hand_pos, target.hand_R.egocentric_hand_pos),
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
            #foot_pos = reward_shape(prop.foot_L.egocentric_foot_pos, target.foot_L.egocentric_foot_pos),
        ),
        leg_L = (
            knee_joint = reward_shape(prop.leg_L.knee_angle, target.leg_L.knee_angle),
            knee_pos = reward_shape(prop.leg_L.egocentric_knee_pos, target.leg_L.egocentric_knee_pos),
            foot_pos = reward_shape(prop.leg_L.egocentric_foot_pos, target.leg_L.egocentric_foot_pos),
            orientation = dot(prop.foot_L.xaxis, target.foot_L.xaxis),
        ),
        foot_R = (
            toe_joint = reward_shape(prop.foot_R.toe_angle, target.foot_R.toe_angle),
            ankle_joint = reward_shape(prop.foot_R.ankle_angle, target.foot_R.ankle_angle),
            orientation = dot(prop.foot_R.xaxis, target.foot_R.xaxis),
            #foot_pos = reward_shape(prop.foot_R.egocentric_foot_pos, target.foot_R.egocentric_foot_pos),
        ),
        leg_R = (
            knee_joint = reward_shape(prop.leg_R.knee_angle, target.leg_R.knee_angle),
            knee_pos = reward_shape(prop.leg_R.egocentric_knee_pos, target.leg_R.egocentric_knee_pos),
            foot_pos = reward_shape(prop.leg_R.egocentric_foot_pos, target.leg_R.egocentric_foot_pos),
            orientation = dot(prop.foot_R.xaxis, target.foot_R.xaxis),
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

reward(env::ModularStandupEnv) = map(sum, compute_rewards(env))

function status(env::ModularStandupEnv)
    if env.randomTrunc[]
        return TRUNCATED
    else
        return RUNNING
    end
end

function info(env::ModularStandupEnv)
    prop = proprioception(env.walker)
    target = env.target
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
        #cumulative_reward = env.cumulative_reward,
        reward_terms = compute_rewards(env),
    )
end

function act!(env::ModularStandupEnv, action)
    set_ctrl!(env.walker, action)
    for _=1:env.walker.n_physics_steps
        step!(env.walker)
    end
    env.lifetime[] += 1
    env.randomTrunc[] = rand() < 0.002
    #env.cumulative_reward .+= reward(env)
end

function reset!(env::ModularStandupEnv)
    env.lifetime[] = 0
    env.randomTrunc[] = false
    reset!(env.walker)
end

function duplicate(env::ModularStandupEnv)
    new_env = ModularStandupEnv(clone(env.walker), env.target, Ref(0), Ref(false))
    reset!(new_env)
    return new_env
end

null_action(env::ModularStandupEnv) = null_action(env.walker)

function Base.show(io::IO, env::ModularStandupEnv)
    #ImLen = im_len(env)
    compact = get(io, :compact, false)
    if compact
        print(io, "ModularStandupEnv{Walker=$(env.walker)}")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)ModularStandupEnv")
        indented_io = IOContext(io, :indent => (get(io, :indent, 0) + 2))
        show(indented_io, env.walker)
        println(io, "$(indent)  lifetime: $(env.lifetime[])")
        #println(io, "$(indent)  cumulative_reward: $(env.cumulative_reward)")
    end
end
