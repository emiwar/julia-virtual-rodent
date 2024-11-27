import CUDA
import Flux
import MPI
import Dates
include("../src/utils/component_tensor.jl") #TODO:
include("../src/environments/mujoco_env.jl")
include("../src/networks/networks.jl")
include("../src/algorithms/mpi_ppo/mpi_ppo.jl")
include("../src/utils/profiler.jl")
include("../src/params.jl")

println("[$(Dates.now())] Initalizing MPI...")
MPI.Init(threadlevel=:funneled)
println("[$(Dates.now()), rank $(MPI.Comm_rank(MPI.COMM_WORLD))] MPI initialized")

template_env = MuJoCoEnvs.RodentImitationEnv(params)
actor_critic = Networks.VariationalEncDec(template_env, params) |> Flux.gpu

n_local_envs = params.rollout.n_envs
envs = [MuJoCoEnvs.clone(template_env, params) for _=1:n_local_envs];

batch_collector = MPI_PPO.BatchCollectorRoot(envs, actor_critic, params)
lapTimer = LapTimer()
batch = batch_collector(lapTimer)
logdict = MPI_PPO.compute_batch_stats(batch)