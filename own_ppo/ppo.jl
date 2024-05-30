import CUDA
import Flux
import Statistics

Base.@kwdef struct ActorCritic{A,C}
    actor::A
    critic::C
end
Flux.@layer ActorCritic

function clamp_logsigma(logsigma, params)
    params.logsigma_min + (params.logsigma_max - params.logsigma_min)*(0.5f0*tanh(logsigma) + 0.5f0)
end

function collect_batch(multi_thread_env, actor_critic, params)
    actionsize = action_size(multi_thread_env)
    statesize = state_size(multi_thread_env)
    nenvs = n_envs(multi_thread_env)
    steps_per_batch = params.n_steps_per_batch

    states =  CUDA.zeros(statesize,  nenvs, steps_per_batch+1)
    actions = CUDA.zeros(actionsize, nenvs, steps_per_batch)
    loglikelihoods = CUDA.zeros(      nenvs, steps_per_batch)
    rewards = CUDA.zeros(             nenvs, steps_per_batch)
    terminated = CUDA.zeros(Bool,     nenvs, steps_per_batch)

    rnd = CUDA.zeros(actionsize, nenvs)
    actor_output = CUDA.zeros(2*actionsize, nenvs)

    prepare_batch!(multi_thread_env, params)
    states[:, :, 1] = multi_thread_env.states
    sum_sigma = 0.0
    for t=1:steps_per_batch
        actor_output .= actor_critic.actor(view(states, :, :, t))
        mu = view(actor_output, 1:actionsize, :)
        #logsigma = (view(actor_output, (actionsize+1):2*actionsize, :))
        sigma = params.sigma_min .+ 0.5f0.*(params.sigma_max .- params.sigma_min).*(1 .+ view(actor_output, (actionsize+1):2*actionsize, :))
        CUDA.randn!(rnd)
        actions[:, :, t] .= mu .+ sigma .* rnd #clamp.(mu .+ sigma .* rnd, -1.0, 1.0) #
        loglikelihoods[:, t] .= (-0.5f0 .* sum(((view(actions, :, :, t) .- mu) ./ sigma).^2; dims=1) .- mapreduce(log, +, sigma; dims=1))'
        
        sum_sigma += sum(sigma)
        multi_thread_env.actions .= cpu(view(actions, :, :, t))
        step!(multi_thread_env, params)
        rewards[:, t] = multi_thread_env.rewards
        terminated[:, t] = multi_thread_env.terminated
        states[:, :, t+1] = multi_thread_env.states
    end
    extra_stats = (;epoch_avg_sigma = sum_sigma / length(actions),
                    avg_sqr_control = mapreduce(a->a^2, +, actions) / length(actions))
    
    return (;states, actions, loglikelihoods, rewards, terminated,
             stats=merge(stats(multi_thread_env), extra_stats))
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
    action_size = size(batch.actions, 1)
    state_values = reshape(batch.states, Val(2)) |> actor_critic.critic
    batch = merge(batch, (;values=reshape(state_values, n_envs, n_steps_per_batch+1)))
    advantages = view(compute_advantages(batch, params), :)
    stats = Dict{Symbol, Float64}()
    gradients = Flux.gradient(actor_critic) do actor_critic
        non_final_states = reshape(view(batch.states, :, :, 1:n_steps_per_batch), Val(2))
        batch_actions = reshape(batch.actions, Val(2))

        actor_output = actor_critic.actor(non_final_states)
        mu = view(actor_output, 1:action_size, :)
        sigma = params.sigma_min .+ 0.5f0.*(params.sigma_max .- params.sigma_min).*(1 .+ view(actor_output, (action_size+1):2*action_size, :))
        new_loglikelihoods = (-0.5f0 .* sum(((batch_actions .- mu) ./ sigma).^2; dims=1) .- sum(log.(sigma); dims=1))
        #logsigma = params.logsigma_min .+ (params.logsigma_max .- params.logsigma_min).*(0.5f0.*tanh.((view(actor_output, (action_size+1):2*action_size, :))) .+ 0.5f0)
        #new_loglikelihoods = -0.5f0 .* sum(((batch_actions .- mu) ./ exp.(logsigma)).^2; dims=1) .- sum(logsigma; dims=1)
        #entropy_loss = mean(action_size * (log(2.0f0*pi) + 1) .+ sum(logsigma; dims = ?)) / 2
        #entropy_loss = (sum(logsigma) + size(logsigma, 1) * (log(2.0f0*pi) + 1)) / (size(logsigma, 2) * 2)
        likelihood_ratios = exp.(view(new_loglikelihoods, :) .- view(batch.loglikelihoods, :))
        grad_cand1 = likelihood_ratios .* advantages
        clamped_ratios = clamp.(likelihood_ratios,
                                1.0f0 - Float32(params.clip_range),
                                1.0f0 + Float32(params.clip_range))
        grad_cand2 = clamped_ratios .* advantages
        actor_loss = -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

        values = view(actor_critic.critic(non_final_states), :)#view(critic_net(non_final_states), :)
        non_final_statevalues = reshape(view(batch.values, :, 1:n_steps_per_batch), Val(1))
        critic_loss = sum((non_final_statevalues .+ advantages .- values).^2) / length(non_final_statevalues)

        total_loss = params.loss_weight_actor * actor_loss + 
                     params.loss_weight_critic * critic_loss# +
                     #params.loss_weight_entropy * entropy_loss
        Flux.ignore() do
            stats[:total_loss] = total_loss
            stats[:critic_loss] = critic_loss
            stats[:actor_loss] = actor_loss
            #stats[:entropy_loss] = entropy_loss
            stats[:epoch_avg_advantage] = Statistics.mean(advantages)
            stats[:epoch_avg_target_value] = Statistics.mean(non_final_statevalues .+ advantages)
            stats[:epoch_avg_predicted_value] = Statistics.mean(values)

            stats[:epoch_std_advantage] = Statistics.std(advantages)
            stats[:epoch_std_target_value] = Statistics.std(non_final_statevalues .+ advantages)
            stats[:epoch_std_predicted_value] = Statistics.std(values)
            stats[:critic_corr] = Statistics.cor(Flux.cpu(values)[:], Flux.cpu(non_final_statevalues .+ advantages)[:])
            stats[:critic_r2] = 1.0 - critic_loss / (stats[:epoch_std_target_value]^2)
        end
        return total_loss
    end
    Flux.update!(opt_state, actor_critic, gradients[1]) #(actor_net, critic_net)
    return stats
end
