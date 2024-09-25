using Flux
using ProgressMeter
import Dates
import BSON
include("environments/rodent_imitation_env.jl")
include("algorithms/collector.jl")
include("algorithms/ppo_loss.jl")
include("algorithms/ppo_networks.jl")
include("utils/wandb_logger.jl")
include("params.jl")

function run_ppo(params)
    test_env = RodentImitationEnv(params)
    actor_critic = ActorCritic(test_env, params) |> Flux.gpu
    opt_state = Flux.setup(Flux.Adam(params.training.learning_rate), actor_critic)
    envs = [clone(test_env, params) for _=1:params.rollout.n_envs]
    starttime = Dates.now()
    run_name = "TestCheckpointLogging-$(starttime)" #ImitationWithAppendages
    lg = Wandb.WandbLogger(project = "Rodent-Imitation", name = run_name, config = params_to_dict(params))
    mkdir("runs/checkpoints/$(run_name)")
    @showprogress for epoch = 1:params.rollout.n_epochs
        batch = collect_batch(envs, actor_critic, params)
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
end

run_ppo(params)