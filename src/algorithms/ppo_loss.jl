import CUDA
import Flux
import Statistics

function compute_advantages(rewards, values, terminateds, params)
    advantages = zero(rewards)
    n_envs, n_steps_per_batch = size(advantages)
    for t in reverse(1:n_steps_per_batch)
        reward = view(rewards, :, t)
        value = view(values, :, t)
        terminated = view(terminateds, :, t)
        next_value = view(values, :, t+1)
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
    old_values = critic(actor_critic, batch.states, params)
    advantages = compute_advantages(batch.rewards, old_values, batch.terminated, params)
    non_final_states = view(batch.states, :, :, 1:n_steps_per_batch)
    non_final_statevalues = view(old_values, :, 1:n_steps_per_batch)
    target_values = non_final_statevalues .+ advantages
    actions = view(batch.actor_output, :action, :, :)
    batch_loglikelihoods = view(batch.actor_output, :loglikelihood, :, :)
    gradients = Flux.gradient(actor_critic) do actor_critic
        #Actor loss
        actor_output = actor(actor_critic, non_final_states, params, actions)
        likelihood_ratios = view(exp.(actor_output.loglikelihood .- batch_loglikelihoods), 1, :, :)
        grad_cand1 = likelihood_ratios .* advantages
        clamped_ratios = clamp.(likelihood_ratios,
                                1.0f0 - Float32(params.clip_range),
                                1.0f0 + Float32(params.clip_range))
        grad_cand2 = clamped_ratios .* advantages
        actor_loss = -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

        #Critic loss
        new_values = critic(actor_critic, non_final_states, params)
        critic_loss = sum((target_values .- new_values).^2) / length(non_final_statevalues)

        #Entropy loss
        sigma = actor_output.sigma#view(actor_output, :sigma, :, :)
        entropy_loss = 0.5sum(log.((2π*exp(1)).*(sigma.^2))) / length(sigma)

        total_loss = params.loss_weight_actor * actor_loss + 
                     params.loss_weight_critic * critic_loss +
                     params.loss_weight_entropy * entropy_loss
        
        Flux.ignore() do
            if !isnothing(logfcn)
                logfcn("losses/total_loss", total_loss)
                logfcn("losses/critic_loss", critic_loss)
                logfcn("losses/actor_loss", actor_loss)
                logfcn("losses/entropy_loss", entropy_loss)
                logfcn("critic/predicted_values", new_values)
                logfcn("critic/target_values", target_values)
                logfcn("critic/advantages", advantages)
                logfcn("critic/explained_variance", 1.0 - critic_loss / Statistics.var(target_values))
                #TODO: this might be quite inefficient
                logfcn("critic/prediction_corr", Statistics.cor(reshape(Array(new_values), :),
                                                                reshape(Array(target_values), :)))
            end
        end
        return total_loss
    end
    Flux.update!(opt_state, actor_critic, gradients[1])
end
