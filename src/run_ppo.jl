import Dates
import MPI
println("[$(Dates.now())] Initalizing MPI...")
MPI.Init(threadlevel=:funneled)
include("environments/rodent_imitation_env.jl")
include("algorithms/collector.jl")
include("params.jl")
if MPI.Comm_rank(MPI.COMM_WORLD)==0
    using Flux
    using ProgressMeter
    import BSON
    include("algorithms/ppo_loss.jl")
    include("algorithms/ppo_networks.jl")
    include("utils/wandb_logger.jl")
end

function run_ppo(params)
    @assert MPI.Is_thread_main()
    mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpi_size = MPI.Comm_size(MPI.COMM_WORLD)
    @assert params.rollout.n_envs % mpi_size == 0
    test_env = RodentImitationEnv(params)
    n_local_envs = params.rollout.n_envs ÷ mpi_size
    
    envs = [clone(test_env, params) for _=1:n_local_envs]

    if mpi_rank == 0
        actor_critic = ActorCritic(test_env, params) |> Flux.gpu
        opt_state = Flux.setup(Flux.Adam(params.training.learning_rate), actor_critic)
        
        starttime = Dates.now()
        run_name = "TestMPI-$(starttime)" #ImitationWithAppendages
        lg = Wandb.WandbLogger(project = "Rodent-Imitation", name = run_name, config = params_to_dict(params))
        mkdir("runs/checkpoints/$(run_name)")
        println("[$(Dates.now())] Root ready...")
        @showprogress for epoch = 1:params.rollout.n_epochs
            batch = collect_batch_root(envs, actor_critic, params)
            ppo_log = ppo_update!(batch, actor_critic, opt_state, params)

            logdict = compute_batch_stats(batch)
            merge!(logdict, ppo_log)
            logdict["total_steps"] = epoch * params.rollout.n_envs * params.rollout.n_steps_per_epoch

            if epoch % params.training.checkpoint_interval == 0
                checkpoint_fn = "runs/checkpoints/$(run_name)/step-$(epoch).bson"
                BSON.bson(checkpoint_fn; actor_critic=Flux.cpu(actor_critic))
                lg.wrun.log_model(checkpoint_fn, "checkpoint-step-$(epoch).bson")
            end
            Wandb.log(lg, logdict)
        end
        Wandb.close(lg);
    else
        println("[$(Dates.now())] Worker $mpi_rank ready...")
        for epoch = 1:params.rollout.n_epochs
            collect_batch_worker(envs, params)
        end
    end
end

run_ppo(params)
MPI.Finalize()