abstract type ImitationRewardSpec end

Base.@kwdef struct EqualRewardWeights <: ImitationRewardSpec
    falloff::NamedTuple{(:com, :rotation, :joint, :joint_vel, :appendages), NTuple{5, Float64}}
    control_cost::Float64
    alive_bonus::Float64
    energy_cost::Float64
end

function compute_rewards(env::ImitationEnv{R, EqualRewardWeights}) where R<:Rodent
    spec = env.reward_spec
    angle_error = angle_to_target(env)
    joint_error_sqr = joint_error(env)
    joint_vel_error_sqr = joint_vel_error(env)
    ctrl = env.walker.data.ctrl
    return (
        com_reward = exp(-(norm(com_error(env))^2) / (spec.falloff.com^2)),
        angle_reward = exp(-(angle_error^2) / (spec.falloff.rotation^2)),
        joint_reward = exp(-joint_error_sqr / (spec.falloff.joint^2)),
        joint_vel_reward = exp(-joint_vel_error_sqr / (spec.falloff.joint_vel^2)),
        append_reward = appendages_reward(env),
        ctrl_penalty = -spec.control_cost * norm(ctrl)^2,
        energy_penalty = -spec.energy_cost * energy_use(env.walker),
        alive_bonus = spec.alive_bonus,
    )
end

function appendages_reward(env::ImitationEnv{R, EqualRewardWeights}) where R<:Rodent
    spec = env.reward_spec
    errors = appendages_error(env)
    reward = 0.0
    for i in axes(errors, 2)
        reward += exp(-norm(@view errors[:, i])^2 / (spec.falloff.appendages^2))
    end
    return reward / size(errors, 2)
end

function Base.show(io::IO, erw::EqualRewardWeights)
    compact = get(io, :compact, false)
    if compact
        print(io, "EqualRewardWeights")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)EqualRewardWeights")
        println(io, "$(indent)  control_cost: ", erw.control_cost)
        println(io, "$(indent)  alive_bonus: ", erw.alive_bonus)
        println(io, "$(indent)  energy_cost: ", erw.energy_cost)
        println(io, "$(indent)  falloff:")
        for k in keys(erw.falloff)
            println(io, "$(indent)    ", k, ": ", erw.falloff[k])
        end
    end
end