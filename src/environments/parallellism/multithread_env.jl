struct MultithreadEnv{S, I, E} <: AbstractEnv
    states::S
    rewards::Vector{Float32}
    status::Vector{UInt8}
    infos::I
    actions::Matrix{Float32}
    environments::Vector{E}
end

function MultithreadEnv(template_env, n_envs)
    template_state  = state(template_env) |> ComponentTensor
    template_info   = info(template_env) |> ComponentTensor
    template_action = null_action(template_env)

    zeros32(inds...) = zeros(Float32, inds...)
    states  = BatchComponentTensor(template_state, n_envs, array_fcn=zeros32)
    rewards = zeros(Float32, n_envs)
    status  = zeros(UInt8, n_envs)
    infos   = BatchComponentTensor(template_info, n_envs, array_fcn=zeros32)
    actions = zeros(Float32, length(template_action), n_envs)

    environments = [duplicate(template_env) for _=1:n_envs]
    MultithreadEnv(states, rewards, status, infos, actions, environments)
end

#Env interface
state(mc::MultithreadEnv)  = mc.states
reward(mc::MultithreadEnv) = mc.rewards
status(mc::MultithreadEnv) = mc.status
info(mc::MultithreadEnv)  = mc.infos

#Utils
raw_states(mc::MultithreadEnv)  = data(mc.states)
raw_infos(mc::MultithreadEnv)   = data(mc.infos)
actions(mc::MultithreadEnv) = mc.actions
n_envs(multithreadEnv::MultithreadEnv) = length(multithreadEnv.environments)
env_type(multithreadEnv::MultithreadEnv) = eltype(multithreadEnv.environments)

function prepare_epoch!(multithreadEnv::MultithreadEnv)
    @Threads.threads for i=1:n_envs(multithreadEnv)
        env = multithreadEnv.environments[i]
        prepare_epoch!(env)
        multithreadEnv.states[:, i] = state(env)
        multithreadEnv.status[i] = status(env)
        multithreadEnv.infos[:, i] = info(env)
    end
end

function act!(multithreadEnv::MultithreadEnv, actions, auto_reset=true)
    @Threads.threads for i=1:n_envs(multithreadEnv)
        env = multithreadEnv.environments[i]
        action = @view actions[:, i]
        act!(env, action)
        multithreadEnv.states[:, i] = state(env)
        multithreadEnv.rewards[i]   = reward(env)
        multithreadEnv.status[i]    = status(env)
        multithreadEnv.infos[:, i]  = info(env)
        if auto_reset && multithreadEnv.status[i] != RUNNING
            reset!(env)
        end
    end
end

