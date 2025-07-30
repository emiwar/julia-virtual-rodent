abstract type MpiEnv end

@concrete struct MpiEnvRoot <: MpiEnv
    states<:ComponentArray
    rewards<:Union{Vector{Float32}, ComponentArray}
    status::Vector{UInt8}
    infos<:ComponentArray
    actions<:Union{Matrix{Float32}, ComponentArray}
    base_env<:AbstractEnv
end

@concrete struct MpiEnvWorker <: MpiEnv
    base_env<:AbstractEnv
end

function MpiEnv(template_env, n_envs; block_workers::Bool, n_steps_per_epoch::Int=-1)
    @assert MPI.Is_thread_main()
    mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpi_size = MPI.Comm_size(MPI.COMM_WORLD)
    if n_envs % mpi_size != 0
        error("n_envs must be a multiple of mpi size ($n_envs % $mpi_size != 0).")
    end
    n_local_envs = n_envs ÷ mpi_size
    base_env = MultithreadEnv(template_env, n_local_envs)
    if mpi_rank == 0
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

        return MpiEnvRoot(states, rewards, status, infos, actions, base_env)
    else
        worker = MpiEnvWorker(base_env)

        #Typically we don't want the non-root processes to do anything else than running
        #`step!` on a loop.
        if block_workers
            if n_steps_per_epoch == -1
                error("`n_steps_per_epoch` must be set when block_workers is true.")
            end
            while true
                prepare_epoch!(worker)
                for t = 1:n_steps_per_epoch
                    step!(worker)
                end
            end
            exit(0)
        end
    end
end

#Env interface
state(mc::MpiEnvRoot)  = mc.states
info(mc::MpiEnvRoot)  = mc.infos
reward(mc::MpiEnv) = mc.rewards
status(mc::MpiEnv) = mc.status
actions(mc::MpiEnv) = mc.actions

raw_states(mc::MpiEnvRoot)  = getdata(mc.states)
raw_infos(mc::MpiEnvRoot)   = getdata(mc.infos)
raw_actions(mc::MpiEnvRoot)   = getdata(mc.actions)
raw_rewards(mc::MpiEnvRoot)   = getdata(mc.rewards)

raw_states(::MpiEnvWorker)  = nothing
raw_infos(::MpiEnvWorker)   = nothing
status(::MpiEnvWorker)  = nothing
reward(::MpiEnvWorker) = nothing
actions(::MpiEnvWorker) = nothing
raw_rewards(::MpiEnvWorker) = nothing
raw_actions(::MpiEnvWorker) = nothing

n_envs(mpiEnv::MpiEnv) = MPI.Comm_size(MPI.COMM_WORLD) * n_envs(mpiEnv.base_env)
env_type(mpiEnv::MpiEnv) = env_type(mpiEnv.localEnv)

function prepare_epoch!(mpiEnv::MpiEnv)
    base_env = mpiEnv.base_env
    prepare_epoch!(base_env)
    MPI.Gather!(raw_states(base_env), raw_states(mpiEnv), MPI.COMM_WORLD)
    MPI.Gather!(status(base_env),  status(mpiEnv), MPI.COMM_WORLD)
    MPI.Gather!(raw_infos(base_env),  raw_infos(mpiEnv), MPI.COMM_WORLD)
end

function act!(mpiEnv::MpiEnvRoot, actions, auto_reset::Bool=true)
    lap(:rollout_action_to_cpu)
    copyto!(mpiEnv.actions, actions)
    step!(mpiEnv, auto_reset)
end

function step!(mpiEnv::MpiEnv, auto_reset::Bool=true)
    base_env = mpiEnv.base_env
    lap(:mpi_scatter_actions)
    MPI.Scatter!(raw_actions(mpiEnv), raw_actions(base_env), MPI.COMM_WORLD)
    lap(:rollout_envs)
    act!(base_env, auto_reset)
    lap(:mpi_wait_for_workers)
    MPI.Barrier(MPI.COMM_WORLD)
    lap(:mpi_gather_state)
    MPI.Gather!(raw_states(base_env),  raw_states(mpiEnv), MPI.COMM_WORLD)
    MPI.Gather!(raw_infos(base_env),  raw_infos(mpiEnv), MPI.COMM_WORLD)
    MPI.Gather!(raw_rewards(base_env),  raw_rewards(mpiEnv), MPI.COMM_WORLD)
    MPI.Gather!(status(base_env),  status(mpiEnv), MPI.COMM_WORLD)
end

function Base.show(io::IO, mpiEnv::MpiEnv)
    workertype(mpiEnv::MpiEnvRoot) = "Root"
    workertype(mpiEnv::MpiEnvWorker) = "Worker"
    compact = get(io, :compact, false)
    if compact
        print(io, "MpiEnv", workertype(mpiEnv))
    else
        println(io, "MpiEnv", workertype(mpiEnv))
        println(io, "\tEnv: $(env_type(mpiEnv))")
        println(io, "\tNum envs: $(n_envs(mpiEnv))")
        println(io, "\tNum processes: $(MPI.Comm_size(MPI.COMM_WORLD))")
        println(io, "\tProcesses rank: $(MPI.Comm_rank(MPI.COMM_WORLD))")
    end
end
