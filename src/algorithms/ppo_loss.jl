import CUDA
import Flux
import Statistics

function compute_advantages(rewards, values, statuses, gamma, lambda)
    advantages = zero(rewards)
    n_envs, n_steps_per_batch = size(advantages)
    for t in reverse(1:n_steps_per_batch)
        reward = view(rewards, :, t)
        value = view(values, :, t)
        status = view(statuses, :, t)
        next_value = next_value_f.(view(values, :, t), view(values, :, t+1), status)
        advantages[:, t] .= reward .+ gamma .* next_value .- value
        if t<n_steps_per_batch
            next_advantage = view(advantages, :, t+1)
            advantages[:, t] .+= gamma .* lambda .* next_advantage .* (status .== RUNNING)
        end
    end
    return advantages
end

function next_value_f(current_state_value, next_state_value, status)
    if status == RUNNING
        return next_state_value
    elseif status == TRUNCATED
        return current_state_value
    else#elseif status==TERMINATED
        return zero(current_state_value)
    end
end

function ppo_update!(batch, actor_critic, opt_state, params)
    println("  [$(Dates.now())] Preparing gradient update...")
    logdict = Dict{String, Float64}()
    n_envs, n_steps_per_batch = size(batch.rewards)
    println("  [$(Dates.now())] Running critic...")
    old_values = critic(actor_critic, batch.states, params)
    println("  [$(Dates.now())] Computing advantages...")
    advantages = compute_advantages(batch.rewards, old_values, batch.status,
                                    params.training.gamma, params.training.lambda)
    non_final_states = view(batch.states, :, :, 1:n_steps_per_batch)
    non_final_statevalues = view(old_values, :, 1:n_steps_per_batch)
    target_values = non_final_statevalues .+ advantages
    actions = view(batch.actor_output, :action, :, :)
    batch_loglikelihoods = view(batch.actor_output, :loglikelihood, :, :)
    clip_range = Float32(params.training.clip_range)
    println("  [$(Dates.now())] Starting miniepochs...")
    for j = params.training.n_miniepochs
        gradients = Flux.gradient(actor_critic) do actor_critic
            #Actor loss
            actor_output = actor(actor_critic, non_final_states, params, actions)
            likelihood_ratios = view(exp.(actor_output.loglikelihood .- batch_loglikelihoods), 1, :, :)
            grad_cand1 = likelihood_ratios .* advantages
            clamped_ratios = clamp.(likelihood_ratios, 1.0f0 - clip_range, 1.0f0 + clip_range)
            grad_cand2 = clamped_ratios .* advantages
            actor_loss = -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

            #Critic loss
            new_values = critic(actor_critic, non_final_states, params)
            critic_loss = sum((target_values .- new_values).^2) / length(non_final_statevalues)

            #Entropy loss
            sigma = actor_output.sigma#view(actor_output, :sigma, :, :)
            entropy_loss = 0.5sum(log.((2π*exp(1)).*(sigma.^2))) / length(sigma)

            total_loss = params.training.loss_weight_actor * actor_loss + 
                        params.training.loss_weight_critic * critic_loss +
                        params.training.loss_weight_entropy * entropy_loss
            
            Flux.ignore() do
                logdict["losses/total_loss"] = total_loss
                logdict["losses/critic_loss"] = critic_loss
                logdict["losses/actor_loss"] = actor_loss
                logdict["losses/entropy_loss"] = entropy_loss
                merge!(logdict, quantile_dict("critic/predicted_values", new_values))
                merge!(logdict, quantile_dict("critic/target_values", target_values))
                merge!(logdict, quantile_dict("critic/advantages", advantages))
                logdict["critic/explained_variance"] = 1.0 - critic_loss / Statistics.var(target_values)
            end
            return total_loss
        end
        Flux.update!(opt_state, actor_critic, gradients[1])
    end
    return logdict
end
