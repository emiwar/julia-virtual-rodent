using Flux
using ProgressMeter
import Dates
import BSON

include("../mujoco_env/rodent_imitation_env.jl")
include("collector.jl")
include("ppo.jl")
include("logger.jl")
include("networks.jl")

params = (;hidden1_size=64,
           hidden2_size=64,
           n_envs=512,
           n_steps_per_batch=16,
           n_physics_steps=5,
           n_miniepochs=5,
           forward_reward_weight = 10.0,
           healthy_reward_weight = 1.0,
           ctrl_reward_weight = 0.0,#30.0,#0.1,
           loss_weight_actor = 1.0,
           loss_weight_critic = 1.0,
           loss_weight_entropy = 0.0,#-0.5,
           min_torso_z = 0.04,
           gamma=0.99,
           lambda=0.95,
           clip_range=0.2,
           n_epochs=50_000,
           sigma_min=1f-2,
           sigma_max=1f0,
           actor_sigma_init_bias=0f0,
           reset_epoch_start=false,
           imitation_steps_ahead=20,
           checkpoint_interval=1000,
           max_target_distance=3e-2,
           reward_sigma_sqr=(1e-2)^2)

test_env = RodentImitationEnv()

actor_critic = ActorCritic(test_env, params) |> Flux.gpu
opt_state = Flux.setup(Flux.Adam(), actor_critic)
envs = [RodentImitationEnv() for _=1:params.n_envs]
run_name = "test-$(Dates.now())"
logger = create_logger("runs/$(run_name).h5", params.n_epochs, 32)
write_params("runs/$(run_name).h5", params)
mkdir("runs/checkpoints/$(run_name)")
@showprogress for epoch = 1:params.n_epochs
    logfcn = (k,v)->logger(epoch, k, v)
    batch = collect_batch(envs, actor_critic, params; logfcn)
    for j=1:(params.n_miniepochs-1)
        ppo_update!(batch, actor_critic, opt_state, params)
    end
    ppo_update!(batch, actor_critic, opt_state, params; logfcn)
    if epoch % params.checkpoint_interval == 0
        BSON.bson("runs/checkpoints/$(run_name)/step-$(epoch).bson"; actor_critic=Flux.cpu(actor_critic))
    end
    GC.gc()
end

