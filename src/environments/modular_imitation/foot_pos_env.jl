#=
Trunk 
* Controls: lumbar, hip for support legs, shoulder for support arms
* Rewards: height, zaxis, pelvis zaxis

Leg:
* Support:
    - Controls: knee + foot
    - Rewards: contact + xaxis (?)

* Not support:
    - Controls: hip + knee + foot
    - Rewards: linear distance to target, no contact

* Switching rule:
    - If closer than 5mm: support
    - If farther than 10mm and at least two supports: not support

Head:
- Always independent
=#

@concrete terse mutable struct FootPosEnv <: AbstractEnv
    walker<:Walker
    target<:ComponentVector
    lifetime::Int64
    randomTrunc::Bool
    supportLimb::SVector{4, Bool}
end

function FootPosEnv(walker)
    reset!(walker)
    target = ComponentVector(proprioception(walker))
    env = FootPosEnv(walker, target, 0, false, (@SVector ones(Bool, 4)))
    reset!(env)
    return env
end

function state(env::FootPosEnv)
    prop = proprioception(env.walker)
    hand_L_target = SVector(0.0, 0.0)
    hand_R_target = SVector(0.0, 0.0)
    foot_L_target = SVector(0.0, 0.0)
    foot_R_target = SVector(0.0, 0.0)
    head_target = SVector{3}(prop.head.egocentric_pos) - SVector{3}(env.target.head.egocentric_pos)
    (
        torso = (
            proprioception = prop.torso,
            support = float.(env.supportLimb),

            hip_L_supinate = prop.leg_L.hip_supinate,
            hip_L_abduct = prop.leg_L.hip_abduct,
            hip_L_extend = prop.leg_L.hip_extend,
            hip_L_force  = prop.leg_L.hip_force,
            hip_R_supinate = prop.leg_R.hip_supinate,
            hip_R_abduct = prop.leg_R.hip_abduct,
            hip_R_extend = prop.leg_R.hip_extend,
            hip_R_force  = prop.leg_R.hip_force,
            
            shoulder_L_sup = prop.arm_L.shoulder_sup,
            shoulder_L = prop.arm_L.shoulder,
            scapula_L_extend = prop.arm_L.scapula_extend,
            scapula_L_abduct = prop.arm_L.scapula_abduct,
            scapula_L_supinate = prop.arm_L.scapula_supinate,
            shoulder_L_force = prop.arm_L.shoulder_force,
            shoulder_R_sup = prop.arm_R.shoulder_sup,
            shoulder_R = prop.arm_R.shoulder,
            scapula_R_extend = prop.arm_R.scapula_extend,
            scapula_R_abduct = prop.arm_R.scapula_abduct,
            scapula_R_supinate = prop.arm_R.scapula_supinate,
            shoulder_R_force = prop.arm_R.shoulder_force,
        ),
        arm_L = (
            supported = float(env.supportLimb[1]),
            proprioception = prop.arm_L,
            hand = (
                finger_angle = prop.hand_L.finger_angle,
                xaxis = prop.hand_L.xaxis,
                contact = prop.hand_L.palm_contact,
            ),
            rel_target = hand_L_target,
        ),
        arm_R = (
            supported = float(env.supportLimb[2]),
            proprioception = prop.arm_R,
            hand = (
                finger_angle = prop.hand_R.finger_angle,
                xaxis = prop.hand_R.xaxis,
                contact = prop.hand_R.palm_contact,
            ),
            rel_target = hand_R_target,
        ),
        leg_L = (
            supported = float(env.supportLimb[3]),
            proprioception = prop.leg_L,
            foot = (
                toe_angle = prop.foot_L.toe_angle,
                xaxis = prop.foot_L.xaxis,
                sole_contact = prop.foot_L.sole_contact,
                heel_contact = prop.foot_L.heel_contact,
            ),
            rel_target = foot_L_target,
        ),
        leg_R = (
            supported = float(env.supportLimb[4]),
            proprioception = prop.leg_R,
            foot = (
                toe_angle = prop.foot_R.toe_angle,
                xaxis = prop.foot_R.xaxis,
                sole_contact = prop.foot_R.sole_contact,
                heel_contact = prop.foot_R.heel_contact,
            ),
            rel_target = foot_R_target,
        ),
        head = (
            proprioception = prop.head,
            rel_target = head_target,
        )
    )
end

function compute_rewards(env::FootPosEnv)
    reward_shape(a, b) = exp(-(norm(a-b)/0.5))#exp(-(norm(a-b)/0.25)^2)
    cosine_dist(a, b) = dot(a, b) / norm(a) / norm(b)
    prop = proprioception(env.walker)
    target = env.target
    (
        torso = (
            orientation_z = cosine_dist(prop.torso.zaxis, target.torso.zaxis),
            pelvis_z = cosine_dist(prop.torso.pelvis_zaxis, target.torso.pelvis_zaxis),
            height_above_ground = reward_shape(prop.torso.height_above_ground, target.torso.height_above_ground),
        ),
        arm_L = env.supportLimb[1] ? float(prop.hand_L.palm_contact[1]>0.0) : -norm(hand_L_target) - float(prop.hand_L.palm_contact[1]>0.0),
        arm_R = env.supportLimb[2] ? float(prop.hand_R.palm_contact[1]>0.0) : -norm(hand_R_target) - float(prop.hand_R.palm_contact[1]>0.0),
        leg_L = env.supportLimb[3] ? float(prop.foot_L.sole_contact[1]>0.0) : -norm(foot_L_target) - float(prop.foot_L.sole_contact[1]>0.0),
        leg_R = env.supportLimb[4] ? float(prop.foot_R.sole_contact[1]>0.0) : -norm(foot_R_target) - float(prop.foot_R.sole_contact[1]>0.0),
        head = (
            orientation_x = cosine_dist(prop.head.xaxis, target.head.xaxis),
            orientation_z = cosine_dist(prop.head.zaxis, target.head.zaxis),
            head_pos = reward_shape(prop.head.egocentric_pos, target.head.egocentric_pos),
            mandible = reward_shape(prop.head.mandible, target.head.mandible),
        )
    )
end

reward(env::FootPosEnv) = map(sum, compute_rewards(env))

function status(env::FootPosEnv)
    prop = proprioception(env.walker)
    if prop.torso.height_above_ground/10.0 < env.walker.min_torso_z
        return TERMINATED
    elseif env.randomTrunc[]
        return TRUNCATED
    else
        return RUNNING
    end
end

function info(env::FootPosEnv)
    prop = proprioception(env.walker)
    target = env.target
    (
        info(env.walker)...,   
        lifetime = float(env.lifetime[]),
        torso_z_dot = prop.torso.zaxis[3],
        torso_height = prop.torso.height_above_ground,
        head_height = prop.head.egocentric_pos[3],
        #cumulative_reward = env.cumulative_reward,
        reward_terms = compute_rewards(env),
    )
end

function set_ctrl!(env::FootPosEnv, action)
    @assert all(a->all(isfinite, a), action)
    walker = env.walker
    walker.actuators.lumbar_extend     = clamp(action.torso[1], -1.0, 1.0)
    walker.actuators.lumbar_bend       = clamp(action.torso[2], -1.0, 1.0)
    walker.actuators.lumbar_twist      = clamp(action.torso[3], -1.0, 1.0)

    walker.actuators.cervical_extend    = clamp(action.head[1], -1.0, 1.0)
    walker.actuators.cervical_bend      = clamp(action.head[2], -1.0, 1.0)
    walker.actuators.cervical_twist     = clamp(action.head[3], -1.0, 1.0)

    walker.actuators.hip_L_supinate     = clamp(action.torso[4], -1.0, 1.0)
    walker.actuators.hip_L_abduct       = clamp(action.torso[5], -1.0, 1.0)
    walker.actuators.hip_L_extend       = clamp(action.torso[6], -1.0, 1.0)
    walker.actuators.knee_L             = clamp(action.leg_L[1], -1.0, 1.0)

    walker.actuators.ankle_L            = clamp(action.leg_L[2], -1.0, 1.0)
    walker.actuators.toe_L              = clamp(action.leg_L[3], -1.0, 1.0)

    walker.actuators.hip_R_supinate     = clamp(action.torso[7], -1.0, 1.0)
    walker.actuators.hip_R_abduct       = clamp(action.torso[8], -1.0, 1.0)
    walker.actuators.hip_R_extend       = clamp(action.torso[9], -1.0, 1.0)
    walker.actuators.knee_R             = clamp(action.leg_R[1], -1.0, 1.0)

    walker.actuators.ankle_R            = clamp(action.leg_R[2], -1.0, 1.0)
    walker.actuators.toe_R              = clamp(action.leg_R[3], -1.0, 1.0)

    walker.actuators.atlas              = clamp(action.head[4], -1.0, 1.0)
    walker.actuators.mandible           = clamp(action.head[5], -1.0, 1.0)

    walker.actuators.scapula_L_supinate = clamp(action.torso[10], -1.0, 1.0)
    walker.actuators.scapula_L_abduct   = clamp(action.torso[11], -1.0, 1.0)
    walker.actuators.scapula_L_extend   = clamp(action.torso[12], -1.0, 1.0)
    walker.actuators.shoulder_L         = clamp(action.torso[13], -1.0, 1.0)
    walker.actuators.shoulder_sup_L     = clamp(action.torso[14], -1.0, 1.0)
    walker.actuators.elbow_L            = clamp(action.arm_L[1], -1.0, 1.0)

    walker.actuators.wrist_L            = clamp(action.arm_L[2], -1.0, 1.0)
    walker.actuators.finger_L           = clamp(action.arm_L[3], -1.0, 1.0)

    walker.actuators.scapula_R_supinate = clamp(action.torso[15], -1.0, 1.0)
    walker.actuators.scapula_R_abduct   = clamp(action.torso[16], -1.0, 1.0)
    walker.actuators.scapula_R_extend   = clamp(action.torso[17], -1.0, 1.0)
    walker.actuators.shoulder_R         = clamp(action.torso[18], -1.0, 1.0)
    walker.actuators.shoulder_sup_R     = clamp(action.torso[19], -1.0, 1.0)
    walker.actuators.elbow_R            = clamp(action.arm_R[1], -1.0, 1.0)

    walker.actuators.wrist_R            = clamp(action.arm_R[2], -1.0, 1.0)
    walker.actuators.finger_R           = clamp(action.arm_R[3], -1.0, 1.0)
end

function act!(env::FootPosEnv, action)
    set_ctrl!(env, action)
    for _=1:env.walker.n_physics_steps
        step!(env.walker)
    end
    env.lifetime += 1
    env.randomTrunc = rand() < 0.002
    #env.cumulative_reward .+= reward(env)
end

function reset!(env::FootPosEnv)
    env.lifetime = 0
    env.randomTrunc = false
    env.supportLimb = @SVector ones(Bool, 4)
    reset!(env.walker)
end

function duplicate(env::FootPosEnv)
    new_env = FootPosEnv(clone(env.walker), env.target, 0, false, SVector(false, false, false, false))
    reset!(new_env)
    return new_env
end

null_action(env::FootPosEnv) = (
    torso  = (@SVector zeros(19)),
    arm_L  = (@SVector zeros(3)),
    arm_R  = (@SVector zeros(3)),
    leg_L  = (@SVector zeros(3)),
    leg_R  = (@SVector zeros(3)),
    head   = (@SVector zeros(5)),
)

function Base.show(io::IO, env::FootPosEnv)
    #ImLen = im_len(env)
    compact = get(io, :compact, false)
    if compact
        print(io, "FootPosEnv{Walker=$(env.walker)}")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)FootPosEnv")
        indented_io = IOContext(io, :indent => (get(io, :indent, 0) + 2))
        show(indented_io, env.walker)
        println(io, "$(indent)  lifetime: $(env.lifetime[])")
        #println(io, "$(indent)  cumulative_reward: $(env.cumulative_reward)")
    end
end
