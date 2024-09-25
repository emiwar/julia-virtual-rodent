using Flux
using ProgressMeter
import Dates
import BSON
include("environments/rodent_run_env.jl")
include("environments/rodent_imitation_env.jl")
include("algorithms/collector.jl")
include("algorithms/ppo_loss.jl")
include("algorithms/ppo_networks.jl")
include("utils/wandb_logger.jl")

params = (;hidden1_size=1024,
           hidden2_size=512,
           n_envs=512,
           n_steps_per_batch=16,
           n_physics_steps=5,
           n_miniepochs=2,
           forward_reward_weight = 10.0,
           healthy_reward_weight = 0.1,
           ctrl_reward_weight = 0.001,#0.025,#30.0,#0.1,
           loss_weight_actor = 1.0,
           loss_weight_critic = 0.5,
           loss_weight_entropy = -0.2,#-0.00,#-0.5,
           min_torso_z = 0.03,
           gamma=0.95,
           lambda=0.95,
           clip_range=0.2,
           n_epochs=100,
           sigma_min=1f-2,
           sigma_max=5f-1,
           actor_sigma_init_bias=0f0,
           reset_epoch_start=false,
           imitation_steps_ahead=5,
           checkpoint_interval=50,
           max_target_distance=2e-1,
           reward_sigma_sqr=(5e-2)^2,
           reward_angle_sigma_sqr=(0.5)^2,
           reward_joint_sigma_sqr=(2.0)^2,
           reward_appendages_sigma_sqr=(0.02)^2,
           latent_dimension=128,
           min_reward=0.0,
           spawn_z_offset=0.01,
           learning_rate=1e-4,
           torque_control=false)

function run_ppo(params)
    test_env = RodentImitationEnv(params)
    actor_critic = ActorCritic(test_env, params) |> Flux.gpu
    opt_state = Flux.setup(Flux.Adam(params.learning_rate), actor_critic)
    envs = [clone(test_env, params) for _=1:params.n_envs]
    starttime = Dates.now()
    run_name = "TestCheckpointLogging-$(starttime)" #ImitationWithAppendages
    lg = Wandb.WandbLogger(project = "Rodent-Imitation",
                           name = run_name,
                           config = Dict(string(k)=>v for (k,v) in pairs(params)))
    mkdir("runs/checkpoints/$(run_name)")
    @showprogress for epoch = 1:params.n_epochs
        batch = collect_batch(envs, actor_critic, params)
        ppo_log = ppo_update!(batch, actor_critic, opt_state, params)

        logdict = compute_batch_stats(batch)
        merge!(logdict, ppo_log)
        logdict["total_steps"] = epoch*params.n_envs*params.n_steps_per_batch

        if epoch % params.checkpoint_interval == 0
            checkpoint_fn = "runs/checkpoints/$(run_name)/step-$(epoch).bson"
            BSON.bson(checkpoint_fn; actor_critic=Flux.cpu(actor_critic))
            lg.wrun.log_model(checkpoint_fn, "checkpoint-step-$(epoch).bson")
        end

        Wandb.log(lg, logdict, commit=true)
    end
    Wandb.close(lg)
end

run_ppo(params)