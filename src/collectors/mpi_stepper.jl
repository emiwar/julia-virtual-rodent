abstract type MpiStepper end

struct MpiStepperRoot{S, I, L} <: MpiStepper
    states::S
    rewards::Vector{Float32}
    status::Vector{UInt8}
    infos::I
    actions::Matrix{Float32}
    localStepper::L
end

struct MpiStepperWorker{L} <: MpiStepper
    localStepper::L
end

function MpiStepper(template_env, n_envs)
    @assert MPI.Is_thread_main()
    mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpi_size = MPI.Comm_size(MPI.COMM_WORLD)
    if n_envs % mpi_size != 0
        error("n_envs must be a multiple of mpi size ($n_envs % $mpi_size != 0).")
    end
    n_local_envs = n_envs ÷ mpi_size
    localStepper = BatchStepper(template_env, n_local_envs)
    if mpi_rank == 0
        template_state  = state(template_env, params) |> ComponentTensor
        template_info   = info(template_env, params) |> ComponentTensor
        template_action = null_action(template_env, params)
    
        zeros32(inds...) = zeros(Float32, inds...)
        states  = BatchComponentTensor(template_state, n_envs, array_fcn=zeros32)
        rewards = zeros(Float32, n_envs)
        status  = zeros(UInt8, n_envs)
        infos   = BatchComponentTensor(template_info, n_envs, array_fcn=zeros32)
        actions = zeros(Float32, length(template_action), n_envs)
        return MpiStepperRoot(states, rewards, status, infos, actions, localStepper)
    else
        return MpiStepperWorker(localStepper)
    end
end

raw_states(mc::MpiStepperRoot)  = data(mc.states)
raw_infos(mc::MpiStepperRoot)   = data(mc.infos)
status(mc::MpiStepperRoot)  = mc.status
rewards(mc::MpiStepperRoot) = mc.rewards
actions(mc::MpiStepperRoot) = mc.actions

raw_states(::MpiStepperWorker)  = nothing
raw_infos(::MpiStepperWorker)   = nothing
status(::MpiStepperWorker)  = nothing
rewards(::MpiStepperWorker) = nothing
actions(::MpiStepperWorker) = nothing

n_envs(mpiStepper::MpiStepper) = n_envs(mpiStepper.localStepper)
env_type(mpiStepper::MpiStepper) = env_type(mpiStepper.localStepper)

function prepareEpoch!(mpiStepper::MpiStepper, params)
    localStepper = mpiStepper.localStepper
    prepareEpoch!(localStepper, params)
    MPI.Gather!(raw_states(localStepper), raw_states(mpiStepper), MPI.COMM_WORLD)
end

function step!(mpiStepper::MpiStepper, params, lapTimer)
    localStepper = mpiStepper.localStepper
    lap(lapTimer, :mpi_scatter_actions)
    MPI.Scatter!(actions(mpiStepper), actions(localStepper), MPI.COMM_WORLD)
    lap(lapTimer, :rollout_envs)
    step!(localStepper, params)
    lap(lapTimer, :mpi_wait_for_workers)
    MPI.Barrier(MPI.COMM_WORLD)
    lap(lapTimer, :mpi_gather_state)
    MPI.Gather!(raw_states(localStepper),  raw_states(mpiStepper), MPI.COMM_WORLD)
    MPI.Gather!(raw_infos(localStepper),  raw_infos(mpiStepper), MPI.COMM_WORLD)
    MPI.Gather!(rewards(localStepper),  rewards(mpiStepper), MPI.COMM_WORLD)
    MPI.Gather!(status(localStepper),  status(mpiStepper), MPI.COMM_WORLD)
end


function Base.show(io::IO, mpiStepper::MpiStepper)
    workertype(mpiStepper::MpiStepperRoot) = "Root"
    workertype(mpiStepper::MpiStepperWorker) = "Worker"
    compact = get(io, :compact, false)
    if compact
        print(io, "MpiStepper", workertype(mpiStepper))
    else
        println(io, "MpiStepper", workertype(mpiStepper))
        println(io, "\tEnv: $(env_type(mpiStepper))")
        println(io, "\tNum envs: $(n_envs(mpiStepper))")
        println(io, "\tNum processes: $(MPI.Comm_size(MPI.COMM_WORLD))")
        println(io, "\tProcesses rank: $(MPI.Comm_rank(MPI.COMM_WORLD))")
    end
end
