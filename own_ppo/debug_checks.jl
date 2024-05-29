include("env.jl")
include("multithread_env.jl")
include("ppo.jl")

using Flux
using ProgressMeter

params = (;hidden1_size=64,
           hidden2_size=64,
           n_envs=64,
           n_steps_per_batch=256,
           n_physics_steps=5,
           forward_reward_weight = 10.0,
           healthy_reward_weight = 1.0,
           ctrl_reward_weight = 0.03,
           loss_weight_actor = 1.0,
           loss_weight_critic = 1.0,
           loss_weight_entropy = -0.1,
           min_torso_z = 0.035,
           gamma=0.99,
           lambda=0.95,
           clip_range=0.2,
           n_epochs=2500,
           actor_sigma_init_bias=-1.0,
           reset_epoch_start=true
)
modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"
multi_thread_env = MultiThreadedMuJoCo{RodentEnv}(MuJoCo.load_model(modelPath), params.n_envs)

actor_bias = [zeros32(action_size(multi_thread_env)); -2*ones32(action_size(multi_thread_env))]
actor_net = Chain(Dense(state_size(multi_thread_env) => params.hidden1_size, relu),
                  Dense(params.hidden1_size => params.hidden2_size, relu),
                  Dense(params.hidden2_size => 2*action_size(multi_thread_env); init=zeros32, bias=actor_bias))

critic_net = Chain(Dense(state_size(multi_thread_env) => params.hidden1_size, relu),
                   Dense(params.hidden1_size => params.hidden2_size, relu),
                   Dense(params.hidden2_size => 1; init=zeros32))

actor_critic = ActorCritic(actor_net, critic_net) |> Flux.gpu
opt_state = Flux.setup(Flux.Adam(), actor_critic)

batch = collect_batch(multi_thread_env, actor_critic, params)
nenvs, n_steps_per_batch = size(batch.rewards)
actionsize = size(batch.actions, 1)
statevalues = reshape(batch.states, Val(2)) |> actor_critic.critic
batch = merge(batch, (;values=reshape(statevalues, nenvs, n_steps_per_batch+1)))
advantages = view(compute_advantages(batch, params), :)
statsdict = Dict{Symbol, Float64}()
#gradients = Flux.gradient(actor_critic) do actor_critic
    non_final_states = reshape(view(batch.states, :, :, 1:n_steps_per_batch), Val(2))
    batch_actions = reshape(batch.actions, Val(2))

    actor_output = actor_critic.actor(non_final_states)
    mu = tanh.(view(actor_output, 1:actionsize, :))
    logsigma = view(actor_output, (actionsize+1):2*actionsize, :)

    new_loglikelihoods = -0.5f0 .* sum(((batch_actions .- mu) ./ exp.(logsigma)).^2; dims=1) .- sum(logsigma; dims=1)
    #entropy_loss = mean(action_size * (log(2.0f0*pi) + 1) .+ sum(logsigma; dims = ?)) / 2
    entropy_loss = (sum(logsigma) + size(logsigma, 1) * (log(2.0f0*pi) + 1)) / (size(logsigma, 2) * 2)

    likelihood_ratios = exp.(view(new_loglikelihoods, :) .- view(batch.loglikelihoods, :))

    grad_cand1 = likelihood_ratios .* advantages
    clamped_ratios = clamp.(likelihood_ratios, 1.0f0 - Float32(params.clip_range), 1.0f0 + Float32(params.clip_range))
    grad_cand2 = clamped_ratios .* advantages
    actor_loss = -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

    values = view(actor_critic.critic(non_final_states), :)#view(critic_net(non_final_states), :)
    non_final_statevalues = reshape(view(batch.values, :, 1:n_steps_per_batch), Val(1))
    critic_loss = sum((non_final_statevalues .+ advantages .- values).^2) / length(non_final_statevalues)

    total_loss = params.loss_weight_actor * actor_loss + 
                    params.loss_weight_critic * critic_loss +
                    params.loss_weight_entropy * entropy_loss
    Flux.ignore() do
        statsdict[:total_loss] = total_loss
        statsdict[:critic_loss] = critic_loss
        statsdict[:actor_loss] = actor_loss
        statsdict[:entropy_loss] = entropy_loss
        statsdict[:epoch_avg_advantage] = Statistics.mean(advantages)
        statsdict[:epoch_avg_target_value] = Statistics.mean(non_final_statevalues .+ advantages)
        statsdict[:epoch_avg_predicted_value] = Statistics.mean(values)

        statsdict[:epoch_std_advantage] = Statistics.std(advantages)
        statsdict[:epoch_std_target_value] = Statistics.std(non_final_statevalues .+ advantages)
        statsdict[:epoch_std_predicted_value] = Statistics.std(values)
        statsdict[:critic_r2] = 1.0 - critic_loss / (statsdict[:epoch_std_target_value]^2)
    end
    return total_loss
end
Flux.update!(opt_state, actor_critic, gradients[1])
#return stats