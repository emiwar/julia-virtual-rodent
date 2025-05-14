abstract type AbstractEncDec end

struct ActorCritic{A, C, AS}
    actor::A
    critic::C
    action_sampler::AS
end

@Flux.layer ActorCritic

function ActorCritic(template_env::Environments.AbstractEnv, params::NamedTuple)
    #TODO: This shouldn't be a constructor but a factory util somewhere
    template_state = ComponentArray(Environments.state(template_env))
    action_size = length(Environments.null_action(template_env))

    state_size = length(template_state)

    actor_sizes = [state_size; params.network.actor_size]
    critic_sizes = [state_size; params.network.critic_size]
                    
    return ActorCritic(actor_sizes, critic_sizes, action_size;
                       actor_type=params.network.actor_type,
                       sigma_min=params.network.sigma_min,
                       sigma_max=params.network.sigma_max,
                       gamma=params.training.gamma)
end

function ActorCritic(actor_sizes::Vector{Int}, critic_sizes::Vector{Int}, action_size::Integer;
                     actor_type=:MLP, sigma_min=0.01, sigma_max=0.5, gamma=nothing)
    if actor_type == :MLP
        actor = Chain((create_layers(Dense, "actor", actor_sizes, tanh)...,
                       mu_and_sigma=Dense(decoder_sizes[end] => 2*action_size, tanh, init=zeros32))...)
    elseif actor_type == :LSTM
        #Q: Should the final layer actually be a Dense layer?
        actor = Chain((create_layers(LSTM, "actor", actor_sizes)...,
                       mu_and_sigma=LSTM(actor_sizes[end] => 2*action_size))...)
    end
    action_sampler = GaussianActionSampler(;sigma_min=sigma_min, sigma_max=sigma_max)

    critic  = Chain((create_layers(Dense, "critic", critic_sizes, tanh)...,
                    V_unscaled = Dense(critic_sizes[end] => 1, init=zeros32),
                    V_scaled = V_unscaled -> V_unscaled ./ eltype(V_unscaled)(1.0-gamma))...)
                    
    return ActorCritic(actor, critic, action_sampler)
end

function actor(actor_critic::ActorCritic, state, reset_mask, action=nothing)
    input = getdata(state)
    actor_output = actor_critic.actor(input)
    action_and_loglikelihood = actor_critic.action_sampler(actor_output, action)
    return action_and_loglikelihood
end

function critic(actor_critic::ActorCritic, state)
    input = getdata(state)
    return view(actor_critic.critic(input), 1, :, :)
end

function checkpoint_latent_state(actor_critic::ActorCritic)
    #TODO: what if the critic is stateful? Should account for that in the PPO algo.
    return (;actor = checkpoint_latent_state(actor_critic.actor),
             critic = checkpoint_latent_state(actor_critic.critic))
end

function restore_latent_state!(actor_critic::ActorCritic, latent_state::NamedTuple)
    restore_latent_state!(actor_critic.actor, latent_state.actor)
    restore_latent_state!(actor_critic.critic, latent_state.critic)
end

function regularization_loss(actor_critic::ActorCritic)
    return regularization_loss(actor_critic.actor) +
           regularization_loss(actor_critic.critic)
end
