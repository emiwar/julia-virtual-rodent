struct LocalCollector{S, I, E}
    states::S
    rewards::Vector{Float32}
    status::Vector{UInt8}
    infos::I
    actions::Matrix{Float32}
    environments::Vector{E}
end

raw_states(mc::LocalCollector)  = data(mc.states)
raw_infos(mc::LocalCollector)   = data(mc.infos)
status(mc::LocalCollector)  = mc.status
rewards(mc::LocalCollector) = mc.rewards
actions(mc::LocalCollector) = mc.actions

function LocalCollector(template_env, n_envs)
    template_state  = state(template_env, params) |> ComponentTensor
    template_info   = info(template_env, params) |> ComponentTensor
    template_action = null_action(template_env, params)

    zeros32(inds...) = zeros(Float32, inds...)
    states  = BatchComponentTensor(template_state, n_envs, array_fcn=zeros32)
    rewards = zeros(Float32, n_envs)
    status  = zeros(UInt8, n_envs)
    infos   = BatchComponentTensor(template_info, n_envs, array_fcn=zeros32)
    actions = zeros(Float32, length(template_action), n_envs)

    envs = [clone(template_env, params) for _=1:n_envs]
    LocalCollector(states, rewards, status, infos, actions, environments)
end

function prepareEpoch!(localCollector::LocalCollector, params)
    @Threads.threads for i=1:n_envs(localCollector)
        if params.rollout.reset_on_epoch_start
            reset!(localCollector.environments[i], params)
        end
        localCollector.states[:, i] = state(localCollector.environments[i], params)
    end
end

function step!(localCollector::LocalCollector, params)
    @Threads.threads for i=1:n_envs(localCollector)
        env = localCollector.environments[i]
        action = view(localCollector.actions, :, i)
        act!(env, action, params)
        localCollector.steps[:, i] = state(env, params)
        localCollector.rewards[i]  = reward(env, params)
        localCollector.status[i]   = status(env, params)
        localCollector.infos[:, i] = info(env, params)
        if localCollector.status[i] != RUNNING
            reset!(env, params)
        end
    end
end
