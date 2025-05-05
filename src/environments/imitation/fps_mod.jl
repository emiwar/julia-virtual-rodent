struct FPSMod{I <: ImitationEnv} <: Environments.AbstractEnv
    base_env::I
    scale_range::Float64
end

reward(env::FPSMod) = reward(env.base_env)
info(env::FPSMod) = info(env.base_env)
status(env::FPSMod) = status(env.base_env)
act!(env::FPSMod, actions) = act!(env.base_env, actions)
reset!(env::FPSMod) = reset!(env.base_env)
null_action(env::FPSMod) = null_action(env.base_env)

function state(env::FPSMod)
    base_state = Environments.state(env.base_env)
    clip_log_rel_speed = log(env.base_env.target_fps/50.0)
    proprioception = merge(base_state.proprioception, (;clip_log_rel_speed ))
    return (
        proprioception = proprioception,
        imitation_target = base_state.imitation_target
    )
end

function duplicate(env::FPSMod{ImitationEnv{W, IRS, ImLen, ImTarget}}) where {W, IRS, ImLen, ImTarget}
    log_lower = log(1/env.scale_range)
    log_upper = log(env.scale_range)
    scale_factor = exp(log_lower + rand()*(log_upper - log_lower))
    target_fps = scale_factor*env.base_env.target_fps
    base_env = env.base_env
    new_base_env = ImitationEnv{W, IRS, ImLen, ImTarget}(
        clone(base_env.walker), base_env.reward_spec, base_env.target,
        base_env.max_target_distance, base_env.restart_on_reset, target_fps,
        Ref(0.0), Ref(0), Ref(0), Ref(0.0)
    )
    reset!(new_base_env)
    return FPSMod(new_base_env, env.scale_range)
end