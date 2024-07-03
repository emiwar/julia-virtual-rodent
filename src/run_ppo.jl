using Flux
using ProgressMeter
import Dates
import BSON
include("environments/rodent_run_env.jl")
include("environments/rodent_imitation_env.jl")
include("algorithms/collector.jl")
include("algorithms/ppo_loss.jl")
include("algorithms/ppo_networks.jl")
include("utils/logger.jl")


params = (;hidden1_size=128,
           hidden2_size=64,
           n_envs=512,
           n_steps_per_batch=16,
           n_physics_steps=5,
           n_miniepochs=5,
           forward_reward_weight = 10.0,
           healthy_reward_weight = 1.0,
           ctrl_reward_weight = 0.005,#0.025,#30.0,#0.1,
           loss_weight_actor = 1.0,
           loss_weight_critic = 1.0,
           loss_weight_entropy = -0.01,#-0.5,
           min_torso_z = 0.04,
           gamma=0.9,
           lambda=0.95,
           clip_range=0.2,
           n_epochs=75_000,
           sigma_min=1f-2,
           sigma_max=1f0,
           actor_sigma_init_bias=0f0,
           reset_epoch_start=false,
           imitation_steps_ahead=3,
           checkpoint_interval=1000,
           max_target_distance=4e-1,
           reward_sigma_sqr=(5e-2)^2,
           reward_angle_sigma_sqr=(0.5)^2)

function run_ppo(params)
    test_env = RodentImitationEnv()
    actor_critic = ActorCritic(test_env, params) |> Flux.gpu
    opt_state = Flux.setup(Flux.Adam(), actor_critic)
    envs = [clone(test_env) for _=1:params.n_envs]
    starttime = Dates.now()
    run_name = "RodentComAndDirImitation-$(starttime)"
    logger = create_logger("runs/$(run_name).h5", params.n_epochs, 32)
    write_params("runs/$(run_name).h5", params)
    write_sysinfo("runs/$(run_name).h5")
    mkdir("runs/checkpoints/$(run_name)")
    @showprogress for epoch = 1:params.n_epochs
        epoch_starttime = Dates.now()
        logfcn = (k,v)->logger(epoch, k, v)
        batch = collect_batch(envs, actor_critic, params)
        for j=1:(params.n_miniepochs-1)
            ppo_update!(batch, actor_critic, opt_state, params)
        end
        ppo_update!(batch, actor_critic, opt_state, params; logfcn)
        if epoch % params.checkpoint_interval == 0
            BSON.bson("runs/checkpoints/$(run_name)/step-$(epoch).bson"; actor_critic=Flux.cpu(actor_critic))
        end
        logfcn("actor/mus", view(batch.actor_output, :mu, :, :))
        logfcn("actor/sigmas", view(batch.actor_output, :sigma, :, :))
        logfcn("actor/action_ctrl", view(batch.actor_output, :action, :, :))
        logfcn("actor/action_ctrl_sum_squared", sum(view(batch.actor_output, :action, :, :).^2; dims=1))
        logfcn("rollout_batch/rewards", batch.rewards)
        logfcn("rollout_batch/failure_rate", sum(batch.terminated) / length(batch.terminated))
        logfcn("rollout_batch/lifespan", view(batch.infos.lifetime, 1, :, :)[Array(batch.terminated)])
        for key in keys(index(batch.infos))
            logfcn("rollout_batch/$key", batch.infos[key])
        end
        epoch_time = (Dates.now() - epoch_starttime).value
        logfcn("timer/epoch_time", epoch_time)
        logfcn("timer/elapsed_time", (Dates.now() - starttime).value)
        logfcn("timer/current_time", Dates.datetime2unix(Dates.now()))
        logfcn("timer/steps_per_second", params.n_envs * params.n_steps_per_batch / epoch_time * 1000.0)
    end
end

run_ppo(params)