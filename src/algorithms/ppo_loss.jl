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

function ppo_update!(batch, actor_critic, opt_state, params, lapTimer::LapTimer;
                     before_miniepoch = ()->nothing)
    lap(lapTimer, :ppo_update_init)
    logdict = Dict{String, Float64}()
    n_envs, n_steps_per_batch = size(batch.rewards)
    lap(lapTimer, :ppo_critic)
    old_values = critic(actor_critic, batch.states, params)
    lap(lapTimer, :ppo_advantages)
    advantages = compute_advantages(batch.rewards, old_values, batch.status,
                                    params.training.gamma, params.training.lambda)
    lap(lapTimer, :ppo_target_values)
    non_final_states = view(batch.states, :, :, 1:n_steps_per_batch)
    non_final_statevalues = view(old_values, :, 1:n_steps_per_batch)
    target_values = non_final_statevalues .+ advantages
    actions = view(batch.actor_outputs, :action, :, :)
    batch_loglikelihoods = view(batch.actor_outputs, :loglikelihood, :, :)
    if has_latent_layer(actor_critic)
        latent_eps = view(batch.actor_outputs, :latent_eps, :, :)
    end
    inv_reset_mask = view(batch.status .== RUNNING, :, 1:n_steps_per_batch)
    clip_range = Float32(params.training.clip_range)
    for j = 1:params.training.n_miniepochs
        lap(lapTimer, :ppo_gradients)
        before_miniepoch()
        gradients = Flux.gradient(actor_critic) do actor_critic
            #Actor loss
            if has_latent_layer(actor_critic)
                actor_output = actor(actor_critic, non_final_states, inv_reset_mask, params, actions, latent_eps)
            else
                actor_output = actor(actor_critic, non_final_states, inv_reset_mask, params, actions)#, latent_eps)
            end
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
         
            #KL loss
            if has_latent_layer(actor_critic)
                latent_mu = actor_output.latent_mu
                latent_logsigma = actor_output.latent_logsigma
                kl_loss = 0.5sum( -2 .* latent_logsigma .- 1 .+ exp.(2 .* latent_logsigma) .+ latent_mu.^2) / length(latent_mu)
                total_loss += params.training.loss_weight_kl * kl_loss 
            end
            
            Flux.ignore() do
                lap(lapTimer, :logging_ppo_stats)
                logdict["losses/total_loss"]   = total_loss
                logdict["losses/critic_loss"]  = critic_loss
                logdict["losses/actor_loss"]   = actor_loss
                logdict["losses/entropy_loss"] = entropy_loss
                if has_latent_layer(actor_critic)
                    logdict["losses/kl_loss"]  = kl_loss
                end
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
#=
function recurrent_ppo_update!(batch, actor_critic, prev_hidden_state, opt_state, params, lapTimer::LapTimer)
    lap(lapTimer, :ppo_update_init)
    logdict = Dict{String, Float64}()
    n_envs, n_steps_per_batch = size(batch.rewards)
    lap(lapTimer, :ppo_critic)
    old_values = critic(actor_critic, batch.states, params)
    lap(lapTimer, :ppo_advantages)
    advantages = compute_advantages(batch.rewards, old_values, batch.status,
                                    params.training.gamma, params.training.lambda)
    lap(lapTimer, :ppo_target_values)
    non_final_states = view(batch.states, :, :, 1:n_steps_per_batch)
    non_final_statevalues = view(old_values, :, 1:n_steps_per_batch)
    target_values = non_final_statevalues .+ advantages
    actions = view(batch.actor_outputs, :action, :, :)
    status  = batch.status
    batch_loglikelihoods = view(batch.actor_outputs, :loglikelihood, :, :)
    if has_latent_layer(actor_critic)
        latent_eps = view(batch.actor_outputs, :latent_eps, :, :)
    end
    clip_range = Float32(params.training.clip_range)
    for j = 1:params.training.n_miniepochs
        lap(lapTimer, :ppo_gradients)
        gradients = Flux.gradient(actor_critic) do actor_critic
            actor_loss = 0f0
            #Actor loss
            for t = 1:n_steps_per_batch
                if has_latent_layer(actor_critic)
                    actor_output = @view actor(actor_critic, non_final_states[:, :, t], params, actions[:, :, t], latent_eps[:, :, t])
                else
                    actor_output = @view actor(actor_critic, non_final_states[:, :, t], params, actions[:, :, t])
                end
                likelihood_ratios = @view exp.(actor_output.loglikelihood .- batch_loglikelihoods[:, t])[1, :]
                grad_cand1 = @view likelihood_ratios .* advantages[:, t]
                clamped_ratios = clamp.(likelihood_ratios, 1.0f0 - clip_range, 1.0f0 + clip_range)
                grad_cand2 = @view clamped_ratios .* advantages[:, t]
                actor_loss += -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

                #Hidden state
                hidden_state = map(x->map(x .* (status[:, t] .== RUNNING)), actor_output.hidden_state)
            end
            
            #Critic loss
            new_values = critic(actor_critic, non_final_states, params)
            critic_loss = sum((target_values .- new_values).^2) / length(non_final_statevalues)

            #Entropy loss
            sigma = actor_output.sigma#view(actor_output, :sigma, :, :)
            entropy_loss = 0.5sum(log.((2π*exp(1)).*(sigma.^2))) / length(sigma)

            total_loss = params.training.loss_weight_actor * actor_loss + 
                         params.training.loss_weight_critic * critic_loss +
                         params.training.loss_weight_entropy * entropy_loss
         
            #KL loss
            if has_latent_layer(actor_critic)
                latent_mu = actor_output.latent_mu
                latent_logsigma = actor_output.latent_logsigma
                kl_loss = 0.5sum( -2 .* latent_logsigma .- 1 .+ exp.(2 .* latent_logsigma) .+ latent_mu.^2) / length(latent_mu)
                total_loss += params.training.loss_weight_kl * kl_loss 
            end
            
            Flux.ignore() do
                lap(lapTimer, :logging_ppo_stats)
                logdict["losses/total_loss"]   = total_loss
                logdict["losses/critic_loss"]  = critic_loss
                logdict["losses/actor_loss"]   = actor_loss
                logdict["losses/entropy_loss"] = entropy_loss
                if has_latent_layer(actor_critic)
                    logdict["losses/kl_loss"]  = kl_loss
                end
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
=#