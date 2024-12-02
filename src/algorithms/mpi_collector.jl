abstract type MpiCollector end

struct RootMpiCollector{S, I, LC} <: MpiCollector
    states::S
    rewards::Vector{Float32}
    status::Vector{UInt8}
    info::I
    actions::Matrix{Float32}
    localCollector::LC
end

struct WorkerMpiCollector{LC} <: MpiCollector
    localCollector::LC
end

function MpiCollector(template_env, n_envs)
    @assert MPI.Is_thread_main()
    mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpi_size = MPI.Comm_size(MPI.COMM_WORLD)
    if n_envs % mpi_size != 0
        error("n_envs must be a multiple of mpi size ($n_envs % $mpi_size != 0).")
    end
    n_local_envs = n_envs ÷ mpi_size
    if mpi_rank == 0
        return RootMpiCollector(template_env, n_envs)
    else
        return WorkerMpiCollector(LocalCollector(n_local_envs))
    end
end

raw_states(mc::RootMpiCollector)  = data(mc.states)
raw_infos(mc::RootMpiCollector)   = data(mc.infos)
status(mc::RootMpiCollector)  = mc.status
rewards(mc::RootMpiCollector) = mc.rewards
actions(mc::RootMpiCollector) = mc.actions

raw_states(mc::WorkerMpiCollector)  = nothing
raw_infos(mc::WorkerMpiCollector)   = nothing
status(mc::WorkerMpiCollector)  = nothing
rewards(mc::WorkerMpiCollector) = nothing
actions(mc::WorkerMpiCollector) = nothing

function prepareEpoch!(mpiCollector::MpiCollector, params)
    localCollector = mpiCollector.localCollector
    prepareEpoch!(localCollector)
    MPI.Gather!(raw_states(localCollector), raw_states(mpiCollector), MPI.COMM_WORLD)
end

function step!(mpiCollector::MpiCollector, params)
    localCollector = mpiCollector.localCollector
    MPI.Scatter!(actions(mpiCollector), actions(localCollector), MPI.COMM_WORLD)
    step!(localCollector)
    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Gather!(raw_states(localCollector),  raw_states(mpiCollector), MPI.COMM_WORLD)
    MPI.Gather!(raw_infos(localCollector),  raw_infos(mpiCollector), MPI.COMM_WORLD)
    MPI.Gather!(rewards(localCollector),  rewards(mpiCollector), MPI.COMM_WORLD)
    MPI.Gather!(status(localCollector),  status(mpiCollector), MPI.COMM_WORLD)
end