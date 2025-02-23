function compute_advantages(rewards, values, statuses, gamma, lambda)
    advantages = zero(rewards)
    n_envs, n_steps_per_batch = size(advantages)
    for t in reverse(1:n_steps_per_batch)
        reward = view(rewards, :, t)
        value = view(values, :, t)
        status = view(statuses, :, t+1)
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

function ppo_update!(batch, actor_critic, actor_critic_start_state, actor_critic_stop_state,
                     opt_state, params, lapTimer::LapTimer)
    lap(lapTimer, :ppo_update_init)
    logdict = Dict{String, Float64}()
    n_envs, n_steps_per_batch = size(batch.rewards)
    lap(lapTimer, :ppo_critic)
    old_values = critic(actor_critic, batch.states)
    lap(lapTimer, :ppo_advantages)
    advantages = compute_advantages(batch.rewards, old_values, batch.status,
                                    params.training.gamma, params.training.lambda)
    lap(lapTimer, :ppo_target_values)
    non_final_states = view(batch.states, :, :, 1:n_steps_per_batch)
    non_final_statevalues = view(old_values, :, 1:n_steps_per_batch)
    target_values = non_final_statevalues .+ advantages
    actions = view(batch.actor_outputs, :action, :, :)
    batch_loglikelihoods = view(batch.actor_outputs, :loglikelihood, :, :)
    reset_mask = view(batch.status, :, 1:n_steps_per_batch) .!= RUNNING
    clip_range = Float32(params.training.clip_range)
    for j = 1:params.training.n_miniepochs
        lap(lapTimer, :restore_latent_state)
        restore_latent_state!(actor_critic, actor_critic_start_state)
        lap(lapTimer, :ppo_gradients)
        gradients = Flux.gradient(actor_critic) do actor_critic
            actor_output = actor(actor_critic, non_final_states, reset_mask; actions)
            likelihood_ratios = view(exp.(actor_output.loglikelihood .- batch_loglikelihoods), 1, :, :)
            grad_cand1 = likelihood_ratios .* advantages
            clamped_ratios = clamp.(likelihood_ratios, 1.0f0 - clip_range, 1.0f0 + clip_range)
            grad_cand2 = clamped_ratios .* advantages
            actor_loss = -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

            #Critic loss
            new_values = critic(actor_critic, non_final_states)
            critic_loss = sum((target_values .- new_values).^2) / length(non_final_statevalues)

            #Entropy loss
            entropy_loss = actor_output.entropy_loss
            regularization_loss = actor_output.regularization_loss

            total_loss = params.training.loss_weight_actor * actor_loss + 
                         params.training.loss_weight_critic * critic_loss +
                         params.training.loss_weight_entropy * entropy_loss +
                         regularization_loss
         
            Flux.ignore() do
                lap(lapTimer, :logging_ppo_stats)
                logdict["losses/total_loss"]   = total_loss
                logdict["losses/critic_loss"]  = critic_loss
                logdict["losses/actor_loss"]   = actor_loss
                logdict["losses/entropy_loss"] = entropy_loss
                logdict["losses/regularization_loss"]  = regularization_loss
                merge!(logdict, quantile_dict("critic/predicted_values", new_values))
                merge!(logdict, quantile_dict("critic/target_values", target_values))
                merge!(logdict, quantile_dict("critic/advantages", advantages))
                logdict["critic/explained_variance"] = 1.0 - critic_loss / Statistics.var(target_values)
            end
            return total_loss
        end
        #Sanity check: after the first miniepoch, the actor_critic should be the same as
        #after the rollout. But this is not true after subsequent miniepochs, because the
        #weights are no longer the same.
        @assert j>1 || checkpoint_latent_state(actor_critic) ≈ actor_critic_stop_state

        Flux.update!(opt_state, actor_critic, gradients[1])
    end
    return logdict
end
