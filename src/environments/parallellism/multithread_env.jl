@concrete struct MultithreadEnv{E} <: AbstractEnv
    states<:ComponentArray
    rewards<:Union{Vector{Float32}, ComponentArray}
    status::Vector{UInt8}
    infos<:ComponentArray
    actions<:Union{Matrix{Float32}, ComponentArray}
    environments::Vector{E}
end

function MultithreadEnv(template_env, n_envs)
    template_state  = state(template_env) |> ComponentVector
    template_info   = info(template_env) |> ComponentVector
    template_action = null_action(template_env)
    template_reward = reward(template_env)
    state_size, = size(template_state)
    info_size, = size(template_info)
    
    states  = ComponentArray(zeros(Float32, state_size, n_envs), (getaxes(template_state)[1], FlatAxis()))

    status  = zeros(UInt8, n_envs)
    infos   = ComponentArray(zeros(Float32, info_size, n_envs), (getaxes(template_info)[1], FlatAxis()))
    
    if template_action isa Union{ComponentArray, NamedTuple}
        template_action = ComponentArray(template_action)
        action_size, = size(template_action)
        actions = ComponentArray(zeros(Float32, action_size, n_envs), (getaxes(template_action)[1], FlatAxis()))
    else
        action_size, = size(template_action)
        actions = zeros(Float32, action_size, n_envs)
    end

    if template_reward isa Union{ComponentArray, NamedTuple}
        template_reward = ComponentArray(template_reward)
        reward_size, = size(template_reward)
        rewards = ComponentArray(zeros(Float32, reward_size, n_envs), (getaxes(template_reward)[1], FlatAxis()))
    else
        @assert template_reward isa Number
        rewards = zeros(Float32, n_envs)
    end

    environments = [duplicate(template_env) for _=1:n_envs]
    MultithreadEnv{typeof(template_env)}(states, rewards, status, infos, actions, environments)
end

#Env interface
state(mc::MultithreadEnv)  = mc.states
reward(mc::MultithreadEnv) = mc.rewards
status(mc::MultithreadEnv) = mc.status
info(mc::MultithreadEnv)  = mc.infos
null_action(mc::MultithreadEnv) = zero(mc.actions)

#Utils
raw_states(mc::MultithreadEnv)  = getdata(mc.states)
raw_infos(mc::MultithreadEnv)   = getdata(mc.infos)
raw_actions(mc::MultithreadEnv) = getdata(mc.actions)
raw_rewards(mc::MultithreadEnv) = getdata(mc.rewards)
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

function act!(multithreadEnv::MultithreadEnv, auto_reset::Bool=true)
    @Threads.threads for i=1:n_envs(multithreadEnv)
        env = multithreadEnv.environments[i]
        action = @view multithreadEnv.actions[:, i]
        act!(env, action)
        multithreadEnv.states[:, i] = state(env)
        if ndims(multithreadEnv.rewards) == 1
            multithreadEnv.rewards[i] = reward(env)
        else
            multithreadEnv.rewards[:, i] = reward(env)
        end
        multithreadEnv.status[i]    = status(env)
        multithreadEnv.infos[:, i]  = info(env)
        if auto_reset && multithreadEnv.status[i] != RUNNING
            reset!(env)
        end
    end
end

function act!(multithreadEnv::MultithreadEnv, actions::AbstractArray, auto_reset::Bool=true)
    copyto!(multithreadEnv.actions, actions)
    act!(multithreadEnv, auto_reset)
end

function Base.show(io::IO, env::MultithreadEnv)
    compact = get(io, :compact, false)
    if compact
        print(io, "MultithreadEnv(base_env=$(env_type(env)))")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)MultithreadEnv with $(n_envs(env)) copies of:")
        indented_io = IOContext(io, :indent => (get(io, :indent, 0) + 2))
        show(indented_io, env.environments[1])
    end
end
