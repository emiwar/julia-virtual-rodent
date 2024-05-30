include("env.jl")
include("multithread_env.jl")
include("ppo.jl")

using Flux
using ProgressMeter
import Wandb
import Dates

params = (;hidden1_size=64,
           hidden2_size=64,
           n_envs=512,
           n_steps_per_batch=8,
           n_physics_steps=5,
           n_miniepochs=5,
           forward_reward_weight = 10.0,
           healthy_reward_weight = 1.0,
           ctrl_reward_weight = 0.25,
           loss_weight_actor = 1.0,
           loss_weight_critic = 1.0,
           loss_weight_entropy = -0.0,
           min_torso_z = 0.035,
           gamma=0.99,
           lambda=0.95,
           clip_range=0.2,
           n_epochs=15_000,
           sigma_min=1f-2,
           sigma_max=1f0,
           actor_sigma_init_bias=0f0,
           reset_epoch_start=false)
modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"
multi_thread_env = MultiThreadedMuJoCo{RodentEnv}(MuJoCo.load_model(modelPath), params.n_envs)

actor_bias = [zeros32(action_size(multi_thread_env));
              params.actor_sigma_init_bias*ones32(action_size(multi_thread_env))]
actor_net = Chain(Dense(state_size(multi_thread_env) => params.hidden1_size, tanh),
                  Dense(params.hidden1_size => params.hidden2_size, tanh),
                  Dense(params.hidden2_size => 2*action_size(multi_thread_env), tanh; init=zeros32, bias=actor_bias))

critic_net = Chain(Dense(state_size(multi_thread_env) => params.hidden1_size, tanh),
                   Dense(params.hidden1_size => params.hidden2_size, tanh),
                   Dense(params.hidden2_size => 1; init=zeros32)) #; 

lg = Wandb.WandbLogger(project = "PPO-julia",
                       name = "testrun-$(Dates.now())",
                       config = Dict(string(k) => v for (k, v) in pairs(params)))
actor_critic = ActorCritic(actor_net, critic_net) |> Flux.gpu
opt_state = Flux.setup(Flux.Adam(), actor_critic)

@showprogress for epoch = 1:params.n_epochs
    epoch_params = params#epoch < 1000 ? merge(params, (;loss_weight_actor=0.0)) : params
    batch = collect_batch(multi_thread_env, actor_critic, epoch_params)
    for j=1:(params.n_miniepochs-1)
        ppo_update!(batch, actor_critic, opt_state, epoch_params)
    end
    losses = ppo_update!(batch, actor_critic, opt_state, epoch_params)
    infodict = merge(Dict(string(k) => v for (k, v) in pairs(batch.stats)),
                     Dict(string(k) => v for (k, v) in losses))
    Wandb.log(lg, infodict)
    GC.gc()
end

Wandb.close(lg)
#adv = Flux.cpu(compute_advantages(batch, params))
#vals = Flux.cpu(batch.values)
#p = Plots.plot()
#for i = 1:32
#    Plots.plot!(p, adv[i, :] .+ 100*i)
#end
#p
#Plots.show()