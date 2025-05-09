struct MultithreadEnv{S, I, E} <: AbstractEnv
    states::S
    rewards::Vector{Float32}
    status::Vector{UInt8}
    infos::I
    actions::Matrix{Float32}
    environments::Vector{E}
end

function MultithreadEnv(template_env, n_envs)
    template_state  = state(template_env) |> ComponentArray
    template_info   = info(template_env) |> ComponentArray
    template_action = null_action(template_env)
    state_size, = size(template_state)
    info_size, = size(template_info)
    action_size, = size(template_action)

    states  = ComponentArray(zeros(Float32, state_size, n_envs), (getaxes(template_state)[1], FlatAxis()))
    rewards = zeros(Float32, n_envs)
    status  = zeros(UInt8, n_envs)
    infos   = ComponentArray(zeros(Float32, info_size, n_envs), (getaxes(template_info)[1], FlatAxis()))
    actions = zeros(Float32, action_size, n_envs)

    environments = [duplicate(template_env) for _=1:n_envs]
    MultithreadEnv(states, rewards, status, infos, actions, environments)
end

#Env interface
state(mc::MultithreadEnv)  = mc.states
reward(mc::MultithreadEnv) = mc.rewards
status(mc::MultithreadEnv) = mc.status
info(mc::MultithreadEnv)  = mc.infos
null_action(mc::MultithreadEnv) = zero(mc.actions)

#Utils
raw_states(mc::MultithreadEnv)  = ComponentArrays.getdata(mc.states)
raw_infos(mc::MultithreadEnv)   = ComponentArrays.getdata(mc.infos)
actions(mc::MultithreadEnv) = mc.actions
n_envs(multithreadEnv::MultithreadEnv) = length(multithreadEnv.environments)
env_type(multithreadEnv::MultithreadEnv) = eltype(multithreadEnv.environments)

function prepare_epoch!(multithreadEnv::MultithreadEnv)
    @Threads.threads for i=1:n_envs(multithreadEnv)
        env = multithreadEnv.environments[i]
        prepare_epoch!(env)
        multithreadEnv.states[:, i] = state(env) |> ComponentArray
        multithreadEnv.status[i] = status(env)   |> ComponentArray
        multithreadEnv.infos[:, i] = info(env)   |> ComponentArray
    end
end

function act!(multithreadEnv::MultithreadEnv, auto_reset::Bool=true)
    @Threads.threads for i=1:n_envs(multithreadEnv)
        env = multithreadEnv.environments[i]
        action = @view multithreadEnv.actions[:, i]
        act!(env, action)
        multithreadEnv.states[:, i] = state(env)  |> ComponentArray
        multithreadEnv.rewards[i]   = reward(env) |> ComponentArray
        multithreadEnv.status[i]    = status(env) |> ComponentArray
        multithreadEnv.infos[:, i]  = info(env)   |> ComponentArray
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
