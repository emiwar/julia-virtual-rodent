
function multiobjective_ppo(collector, networks, params; logger = (_, _)->nothing)
    opt_state = Flux.setup(Flux.Adam(params.training.learning_rate), networks)
    
    #Main training loop
    @showprogress for epoch = 1:params.rollout.n_epochs
        
        #Checkpoint the latent state of the networks before starting a batch of rollouts
        network_state_at_epoch_start = Networks.checkpoint_latent_state(networks)

        #Collect a batch of rollout data
        collect_epoch!(collector, networks)

        #Checkpoint the latent state of the networks after collecting a batch of rollouts
        #This is just for sanity checking that collector rollouts and gradients rollouts
        #leave the networks in the same latent state.
        network_state_at_epoch_end = Networks.checkpoint_latent_state(networks)

        #Apply the gradient updates
        ppo_log = multiobjective_ppo_update!(collector, networks, network_state_at_epoch_start,
                                             network_state_at_epoch_end, opt_state;
                                             gamma = params.training.gamma,
                                             lambda = params.training.lambda,
                                             clip_range = params.training.clip_range |> Float32,
                                             n_miniepochs = params.training.n_miniepochs,
                                             loss_weights = (actor   = params.training.loss_weight_actor,
                                                             critic  = params.training.loss_weight_critic,
                                                             entropy = params.training.loss_weight_entropy))

        #Compute batch statistics for logging
        logdict = compute_multi_batch_stats(collector)
        merge!(logdict, ppo_log)
        logdict["total_steps"] = epoch * n_envs(collector) * steps_per_epoch(collector)
        logger(epoch, logdict)
    end

end

function multiobjective_ppo_update!(batch, actor_critic, actor_critic_start_state, actor_critic_stop_state,
                                    opt_state; gamma, lambda, clip_range::Float32, n_miniepochs, loss_weights::NamedTuple)
    lap(:ppo_update_init)
    logdict = Dict{String, Float64}()
    n_rewards, n_envs, n_steps_per_batch = size(batch.rewards)
    lap(:ppo_critic)
    old_values = vcat(Networks.critic(actor_critic, batch.states)...)
    lap(:ppo_advantages)
    advantages = compute_multi_advantages(batch.rewards, old_values, batch.status, gamma, lambda)
    lap(:ppo_target_values)
    non_final_states = view(batch.states, :, :, 1:n_steps_per_batch)
    non_final_statevalues = view(old_values, :, :, 1:n_steps_per_batch)
    target_values = getdata(non_final_statevalues .+ advantages)
    actions = batch.actor_outputs.action
    batch_loglikelihoods = sum(getdata(batch.actor_outputs.loglikelihood); dims=1)
    reset_mask = view(batch.status, :, 1:n_steps_per_batch) .!= Environments.RUNNING
    actor_advantages = sum(advantages, dims=1) ./ size(advantages, 1)
    for j = 1:n_miniepochs
        lap(:restore_latent_state)
        Networks.restore_latent_state!(actor_critic, actor_critic_start_state)
        lap(:ppo_gradients)
        gradients = Flux.gradient(actor_critic) do actor_critic
            actor_output = Networks.actor(actor_critic, non_final_states, reset_mask, actions)
            new_likelihoods = sum(actor_output.loglikelihood)
            likelihood_ratios = exp.(new_likelihoods .- batch_loglikelihoods)
            grad_cand1 = likelihood_ratios .* actor_advantages
            clamped_ratios = clamp.(likelihood_ratios, 1.0f0 - clip_range, 1.0f0 + clip_range)
            grad_cand2 = clamped_ratios .* actor_advantages
            actor_loss = -sum(min.(grad_cand1, grad_cand2)) / length(grad_cand1)

            #Critic loss
            new_values = vcat(Networks.critic(actor_critic, non_final_states)...)
            critic_loss = sum((target_values .- new_values).^2) / length(non_final_statevalues)

            #Entropy loss
            entropy_loss = sum(actor_output.entropy_loss) / length(actor_output.entropy_loss)

            total_loss = loss_weights.actor * actor_loss + 
                         loss_weights.critic * critic_loss +
                         loss_weights.entropy * entropy_loss +
                         Networks.regularization_loss(actor_critic)

            Flux.ignore() do
                lap(:logging_ppo_stats)
                logdict["losses/total_loss"]   = total_loss
                logdict["losses/critic_loss"]  = critic_loss
                logdict["losses/actor_loss"]   = actor_loss
                logdict["losses/entropy_loss"] = entropy_loss
                logdict["losses/regularization_loss"]  = Networks.regularization_loss(actor_critic)
                for (i, k) in enumerate(keys(first(getaxes(advantages))))
                    new_v = view(new_values, i, :, :)
                    target_v =  view(target_values, i, :, :)
                    merge!(logdict, quantile_dict("critic/$k/predicted_values", new_v))
                    merge!(logdict, quantile_dict("critic/$k/target_values", target_v))
                    #merge!(logdict, quantile_dict("critic/advantages", advantages))
                    
                    err = sum((target_v .- new_v).^2) / length(new_v)
                    logdict["critic/$k/error"] = err
                    var = sum(target_v.^2) / length(target_v) - (sum(target_v) / length(target_v))^2
                    logdict["critic/$k/unexplained_variance"] = err / var
                end
            end
            return total_loss
        end

        #Sanity check: after the first miniepoch, the actor_critic should be the same as
        #after the rollout. Note however that this is not true after subsequent miniepochs,
        #because the weights are no longer the same.
        @assert j>1 || Networks.checkpoint_latent_state(actor_critic) ≈ actor_critic_stop_state

        #Update the actor_critic weights using the gradients
        Flux.update!(opt_state, actor_critic, gradients[1])
    end
    return logdict
end