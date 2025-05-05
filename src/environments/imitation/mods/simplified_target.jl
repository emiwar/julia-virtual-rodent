struct SimplifiedTarget{I <: ImitationEnv} <: Environments.AbstractEnv
    base_env::I
end

reward(env::SimplifiedTarget) = reward(env.base_env)
info(env::SimplifiedTarget) = info(env.base_env)
status(env::SimplifiedTarget) = status(env.base_env)
act!(env::SimplifiedTarget, actions) = act!(env.base_env, actions)
reset!(env::SimplifiedTarget) = reset!(env.base_env)
null_action(env::SimplifiedTarget) = null_action(env.base_env)
duplicate(env::SimplifiedTarget) = SimplifiedTarget(duplicate(env.base_env))

function state(env::SimplifiedTarget)
    (
        proprioception = proprioception(env.base_env.walker),
        imitation_target = (
            com = reshape(com_horizon(env.base_env), :),
            root_quat = reshape(root_quat_horizon(env.base_env), :),
        )
    )
end
