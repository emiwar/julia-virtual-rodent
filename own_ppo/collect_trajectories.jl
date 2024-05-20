import CUDA
using Flux
include("env.jl")

model = m

params = (;hidden1_size=64,
           hidden2_size=64,
           n_envs=32,
           n_steps_per_batch=128,
           n_physics_steps=5,
           forward_reward_weight = 10.0,
           healthy_reward_weight = 1.0,
           ctrl_reward_weight = 0.1,
           loss_weight_actor = 1.0,
           loss_weight_critic = 1.0,
           loss_weight_entropy = 1.0,
           min_torso_z = 0.035,
           gamma=0.99,
           lambda=0.95,
           clip_range=0.2)

Base.@kwdef struct ActorCritic{A,C}
    actor::A
    critic::C
end

Flux.@layer ActorCritic

action_size = m.nu
state_size = MuJoCo.mj_stateSize(m, MuJoCo.LibMuJoCo.mjSTATE_PHYSICS)

actor_net = Chain(Dense(state_size => params.hidden1_size, relu),
                  Dense(params.hidden1_size => params.hidden2_size, relu),
                  Dense(params.hidden2_size => 2*action_size, relu))

critic_net = Chain(Dense(state_size => params.hidden1_size, relu),
                   Dense(params.hidden1_size => params.hidden2_size, relu),
                   Dense(params.hidden2_size => 1, relu))

actor_critic = ActorCritic(actor_net, critic_net) |> Flux.gpu

envStates = [EnvState(model) for _=1:params.n_envs]

function collect_batch(model, envStates, actor_critic, action_size, state_size, params)
    states = CUDA.zeros(state_size, params.n_envs, params.n_steps_per_batch+1)
    actions = CUDA.zeros(action_size, params.n_envs, params.n_steps_per_batch)
    loglikelihoods = CUDA.zeros(params.n_envs, params.n_steps_per_batch)
    rewards = CUDA.zeros(params.n_envs, params.n_steps_per_batch)
    terminated = CUDA.zeros(Bool, params.n_envs, params.n_steps_per_batch)

    for i=1:params.n_envs
        states[:, i, 1] = state(model, envStates[i])
    end

    actions_cpu = zeros(Float32, action_size, params.n_envs)
    states_cpu = zeros(Float32, state_size, params.n_envs)
    rewards_cpu = zeros(Float32, params.n_envs)
    terminated_cpu = zeros(Bool, params.n_envs)
    t = 1
    for t=1:params.n_steps_per_batch
        action_params = actor_critic.actor(view(states, :, :, t))
        mu = view(action_params, 1:action_size, :)
        logsigma = view(action_params, (action_size+1):2*action_size, :)
        rnd = CUDA.randn(action_size, params.n_envs)
        loglikelihoods[:, t] .= -0.5 .* sum(rnd .^ 2; dims=1)'# .+ (action_size * log(2*pi))
        actions[:, :, t] .= clamp.(mu .+ exp.(logsigma) .* rnd, -1.0f0, 1.0f0)
        actions_cpu .= cpu(view(actions, :, :, t))
        @Threads.threads for i=1:params.n_envs
            act!(model, envStates[i], actions_cpu[:, i], params)
            states_cpu[:, i] .= state(model, envStates[i])
            rewards_cpu[i] = reward(model, envStates[i], params)
            terminated_cpu[i] = is_terminated(envStates[i], params)
            if terminated_cpu[i]
                reset!(model, envStates[i])
            end
        end
        rewards[:, t] = rewards_cpu
        terminated[:, t] = terminated_cpu
        states[:, :, t+1] = states_cpu
    end
    return (;states, actions, loglikelihoods, rewards, terminated)
end

function compute_advantages(batch, params)
    advantages = zero(batch.rewards)
    n_envs, n_steps_per_batch = size(advantages)
    for t in reverse(1:n_steps_per_batch)
        reward = view(batch.rewards, :, t)
        value = view(batch.values, :, t)
        terminated = view(batch.terminated, :, t)
        next_value = view(batch.values, :, t+1)
        advantages[:, t] .= reward .+ params.gamma .* next_value .* (.!terminated) .- value
        if t<n_steps_per_batch
            next_advantage = view(advantages, :, t+1)
            advantages[:, t] .+= params.gamma .* params.lambda .* (.!terminated) .* next_advantage
        end
    end
    return advantages
end

function ppo_update!(batch, actor_critic, opt_state, params)
    n_envs, n_steps_per_batch = size(batch.rewards)
    state_values = reshape(batch.states, Val(2)) |> actor_critic.critic
    batch = merge(batch, (;values=reshape(state_values, n_envs, n_steps_per_batch+1)))
    advantages = view(compute_advantages(batch, params), :)
    
    #weights = Flux.params(actor_net, critic_net)
    gradients = Flux.gradient(actor_critic) do actor_critic
        non_final_states = reshape(view(batch.states, :, :, 1:n_steps_per_batch), Val(2))
        
        action_params = actor_critic.actor(non_final_states)
        mu = view(action_params, 1:action_size, :)
        logsigma = view(action_params, (action_size+1):2*action_size, :)
        new_loglikelihoods = -0.5 .* sum(((reshape(batch.actions, Val(2)) .- mu) ./ exp.(logsigma)).^2; dims=1)
        #entropy_loss = mean(action_size * (log(2.0f0*pi) + 1) .+ sum(logsigma; dims = ?)) / 2
        entropy_loss = (sum(logsigma) + size(logsigma, 1) * (log(2.0f0*pi) + 1)) / (size(logsigma, 2) * 2)

        likelihood_ratios = exp.(view(new_loglikelihoods, :) .- view(batch.loglikelihoods, :))

        grad_cand1 = likelihood_ratios .* advantages
        clamped_ratios = clamp.(likelihood_ratios, 1.0f0 - params.clip_range, 1.0f0 + params.clip_range)
        grad_cand2 = clamped_ratios .* advantages
        actor_loss = -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

        values = view(actor_critic.critic(non_final_states), :)#view(critic_net(non_final_states), :)
        non_final_statevalues = reshape(view(batch.values, :, 1:n_steps_per_batch), Val(1))
        critic_loss = sum((non_final_statevalues .+ advantages .- values).^2) / length(non_final_statevalues)

        println("Critic loss: $critic_loss")
        println("Actor loss: $actor_loss")
        println("entropy loss: $entropy_loss")

        total_loss = params.loss_weight_actor * actor_loss + 
                     params.loss_weight_critic * critic_loss +
                     params.loss_weight_entropy * entropy_loss
        return total_loss
    end
    Flux.update!(opt_state, actor_critic, gradients[1]) #(actor_net, critic_net)
end


opt_state = Flux.setup(Flux.Adam(), actor_critic)
batch = collect_batch(model, envStates, actor_critic, action_size, state_size, params)
for i=1:100
    ppo_update!(batch, actor_critic, opt_state, params);
end

adv = Flux.cpu(compute_advantages(batch, params))
vals = Flux.cpu(batch.values)
p = Plots.plot()
for i = 1:32
    Plots.plot!(p, adv[i, :] .+ 100*i)
end
p
Plots.show()