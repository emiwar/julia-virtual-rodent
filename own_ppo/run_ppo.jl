include("mujoco_env.jl")
include("collector.jl")
include("ppo.jl")
include("logger.jl")
include("networks.jl")


using Flux
using ProgressMeter
import Dates

params = (;hidden1_size=64,
           hidden2_size=64,
           n_envs=512,
           n_steps_per_batch=16,
           n_physics_steps=5,
           n_miniepochs=5,
           forward_reward_weight = 10.0,
           healthy_reward_weight = 1.0,
           ctrl_reward_weight = 0.1,
           loss_weight_actor = 1.0,
           loss_weight_critic = 1.0,
           loss_weight_entropy = -0.0,
           min_torso_z = 0.035,
           gamma=0.99,
           lambda=0.95,
           clip_range=0.2,
           n_epochs=50_000,
           sigma_min=1f-2,
           sigma_max=1f0,
           actor_sigma_init_bias=0f0,
           reset_epoch_start=false)

test_env = RodentEnv()

actor_critic = ActorCritic(test_env, params) |> Flux.gpu
opt_state = Flux.setup(Flux.Adam(), actor_critic)
envs = [RodentEnv() for _=1:params.n_envs]
logger = create_logger("runs/test-$(Dates.now()).h5", params.n_epochs, 32)

@showprogress for epoch = 1:params.n_epochs
    epoch_params = params#epoch < 1000 ? merge(params, (;loss_weight_actor=0.0)) : params
    logfcn = (k,v)->logger(epoch, k, v)
    batch = collect_batch(envs, actor_critic, epoch_params; logfcn)
    for j=1:(params.n_miniepochs-1)
        ppo_update!(batch, actor_critic, opt_state, epoch_params)
    end
    ppo_update!(batch, actor_critic, opt_state, epoch_params; logfcn)
    GC.gc()
end

batch = collect_batch(envs, actor_critic, params);
ppo_update!(batch, actor_critic, opt_state, params);
#Wandb.close(lg)
#adv = Flux.cpu(compute_advantages(batch, params))
#vals = Flux.cpu(batch.values)
#p = Plots.plot()
#for i = 1:32
#    Plots.plot!(p, adv[i, :] .+ 100*i)
#end
#p
#Plots.show()