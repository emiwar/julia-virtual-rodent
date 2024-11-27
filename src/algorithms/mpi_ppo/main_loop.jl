function ppo_training(template_env::Main.MuJoCoEnvs.MuJoCoEnv, actor_critic::Main.Networks.PPOActorCritic,
                      params, run_name_prefix, wandb_project="Rodent-Imitation", epoch_start=1)
    @assert Main.MPI.Is_thread_main()
    mpi_rank = Main.MPI.Comm_rank(MPI.COMM_WORLD)
    mpi_size = Main.MPI.Comm_size(MPI.COMM_WORLD)
    @assert params.rollout.n_envs % mpi_size == 0

    n_local_envs = params.rollout.n_envs ÷ mpi_size
    
    envs = [clone(example_env, params) for _=1:n_local_envs];

    if mpi_rank == 0
        opt_state = Flux.setup(Flux.Adam(params.training.learning_rate), actor_critic)
        batch_collector = BatchCollectorRoot(envs, actor_critic, params)
        starttime = Dates.now()
        run_name = "$(run_name_prefix)-$(starttime)"
        lg = Wandb.WandbLogger(project = wandb_project, name = run_name,
                               config = params_to_dict(params))
        mkdir("runs/checkpoints/$(run_name)")
        println("[$(Dates.now())] Root ready...")
        @showprogress for epoch = epoch_start:params.rollout.n_epochs
            lapTimer = LapTimer()
            batch = batch_collector(lapTimer)
            ppo_log = ppo_update!(batch, actor_critic, opt_state, params, lapTimer)
            lap(lapTimer, :logging_batch_stats)
            logdict = compute_batch_stats(batch)
            merge!(logdict, ppo_log)
            logdict["total_steps"] = epoch * params.rollout.n_envs * params.rollout.n_steps_per_epoch
            lap(lapTimer, :checkpointing)
            if epoch % params.training.checkpoint_interval == 0
                checkpoint_fn = "runs/checkpoints/$(run_name)/step-$(epoch).bson"
                BSON.bson(checkpoint_fn; actor_critic=Flux.cpu(actor_critic))
                lg.wrun.log_model(checkpoint_fn, "checkpoint-step-$(epoch).bson")
            end
            lap(lapTimer, :logging_submitting)
            merge!(logdict, to_stringdict(lapTimer))
            Wandb.log(lg, logdict)
        end
        Wandb.close(lg);
    else
        batch_collector = BatchCollectorWorker(envs, params)
        println("[$(Dates.now())] Worker $mpi_rank ready...")
        for epoch = epoch_start:params.rollout.n_epochs
            batch_collector()
        end
    end
end

function continue_ppo_training(template_env; wandb_id, wandb_project="Rodent-Imitation")
    params = load_params_from_wandb(continue_from)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        actor_critic, epoch = load_actor_critic_from_wandb(continue_from)
    else
        actor_critic = nothing
	    epoch = 1
    end
    ppo_training(template_env, actor_critic, params, "Continue-$(wandb_id)",
                 "Rodent-Imitation", epoch)
end
