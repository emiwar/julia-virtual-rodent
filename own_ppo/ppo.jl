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

function collect_batch(multi_thread_env, actor_critic, params; logfcn=nothing)
    actionsize = action_size(multi_thread_env)
    statesize = state_size(multi_thread_env)
    nenvs = n_envs(multi_thread_env)
    steps_per_batch = params.n_steps_per_batch

    states =  CUDA.zeros(statesize,  nenvs, steps_per_batch+1)
    actions = CUDA.zeros(actionsize, nenvs, steps_per_batch)
    loglikelihoods = CUDA.zeros(     nenvs, steps_per_batch)
    rewards = CUDA.zeros(            nenvs, steps_per_batch)
    terminated = CUDA.zeros(Bool,    nenvs, steps_per_batch)
    sigmas = CUDA.zeros(actionsize,  nenvs, steps_per_batch)
    mus =    CUDA.zeros(actionsize,  nenvs, steps_per_batch)

    rnd = CUDA.zeros(actionsize, nenvs)
    actor_output = CUDA.zeros(2*actionsize, nenvs)
    info = zeros(Float32, length(info_fcns(eltype(multi_thread_env.envs))), nenvs, steps_per_batch)

    prepare_batch!(multi_thread_env, params)
    states[:, :, 1] = multi_thread_env.states
    for t=1:steps_per_batch
        actor_output .= actor_critic.actor(view(states, :, :, t))
        mus[:, :, t] .= view(actor_output, 1:actionsize, :)
        unscaled_sigma = view(actor_output, (actionsize+1):2*actionsize, :)
        sigmas[:, :, t] .= params.sigma_min .+ 0.5f0.*(params.sigma_max .- params.sigma_min).*(1 .+ unscaled_sigma)
        CUDA.randn!(rnd)
        actions[:, :, t] .= view(mus, :, :, t) .+ view(sigmas, :, :, t) .* rnd #clamp.(mu .+ sigma .* rnd, -1.0, 1.0) #
        loglikelihoods[:, t] .= (-0.5f0 .* sum(((view(actions, :, :, t) .- view(mus, :, :, t)) ./ view(sigmas, :, :, t)).^2; dims=1) .- mapreduce(log, +, view(sigmas, :, :, t); dims=1))'
        
        multi_thread_env.actions .= cpu(view(actions, :, :, t))
        step!(multi_thread_env, params)
        rewards[:, t] = multi_thread_env.rewards
        terminated[:, t] = multi_thread_env.terminated
        states[:, :, t+1] = multi_thread_env.states
        info[:, :, t] = multi_thread_env.info_array
    end
    if !isnothing(logfcn)
        logfcn("actor/mus", mus)
        logfcn("actor/sigmas", sigmas)
        logfcn("actor/actions", actions)
        logfcn("actor/squared_actions", actions.^2)
        logfcn("rollout_batch/rewards", rewards)
        for (j, key) in enumerate(keys(info_fcns(eltype(multi_thread_env.envs))))
            logfcn("rollout_batch/$key", info[j, :, :])
        end
        #for (key, val) in pairs(stats(multi_thread_env))
        #    logfcn("rollout_batch/$key", val)
        #end
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

function ppo_update!(batch, actor_critic, opt_state, params; logfcn=nothing)
    n_envs, n_steps_per_batch = size(batch.rewards)
    action_size = size(batch.actions, 1)
    state_values = reshape(batch.states, Val(2)) |> actor_critic.critic
    batch = merge(batch, (;values=reshape(state_values, n_envs, n_steps_per_batch+1)))

    advantages = view(compute_advantages(batch, params), :)
    non_final_states = reshape(view(batch.states, :, :, 1:n_steps_per_batch), Val(2))
    batch_actions = reshape(batch.actions, Val(2))
    non_final_statevalues = reshape(view(batch.values, :, 1:n_steps_per_batch), Val(1))
    target_values = non_final_statevalues .+ advantages

    #stats = Dict{Symbol, Float64}()
    gradients = Flux.gradient(actor_critic) do actor_critic
        #Actor loss
        actor_output = actor_critic.actor(non_final_states)
        mu = view(actor_output, 1:action_size, :)
        unscaled_sigma = view(actor_output, (action_size+1):2*action_size, :)
        sigma = params.sigma_min .+ 0.5f0.*(params.sigma_max .- params.sigma_min).*(1 .+ unscaled_sigma)
        new_loglikelihoods = (-0.5f0 .* sum(((batch_actions .- mu) ./ sigma).^2; dims=1) .- sum(log.(sigma); dims=1))
        likelihood_ratios = exp.(view(new_loglikelihoods, :) .- view(batch.loglikelihoods, :))
        grad_cand1 = likelihood_ratios .* advantages
        clamped_ratios = clamp.(likelihood_ratios,
                                1.0f0 - Float32(params.clip_range),
                                1.0f0 + Float32(params.clip_range))
        grad_cand2 = clamped_ratios .* advantages
        actor_loss = -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

        #Critic loss
        values = view(actor_critic.critic(non_final_states), :)
        critic_loss = sum((target_values .- values).^2) / length(non_final_statevalues)

        #Entropy loss
        #entropy_loss = mean(action_size * (log(2.0f0*pi) + 1) .+ sum(logsigma; dims = ?)) / 2
        #entropy_loss = (sum(logsigma) + size(logsigma, 1) * (log(2.0f0*pi) + 1)) / (size(logsigma, 2) * 2)

        total_loss = params.loss_weight_actor * actor_loss + 
                     params.loss_weight_critic * critic_loss# +
                     #params.loss_weight_entropy * entropy_loss
        
        Flux.ignore() do
            if !isnothing(logfcn)
                logfcn("losses/total_loss", total_loss)
                logfcn("losses/critic_loss", critic_loss)
                logfcn("losses/actor_loss", actor_loss)
                logfcn("critic/predicted_values", values)
                logfcn("critic/target_values", target_values)
                logfcn("critic/advantages", advantages)
                logfcn("critic/prediction_corr", Statistics.cor(Flux.cpu(values)[:], Flux.cpu(target_values)[:]))
                logfcn("critic/explained_variance", 1.0 - critic_loss / Statistics.var(target_values))
            end
        end
        return total_loss
    end
    Flux.update!(opt_state, actor_critic, gradients[1]) #(actor_net, critic_net)
    #return stats
end
