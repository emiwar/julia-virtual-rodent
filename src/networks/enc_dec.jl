abstract type AbstractEncDec end

struct EncDec{E, D, C, A} <: AbstractEncDec
    encoder::E
    decoder::D
    critic::C
    action_sampler::A
end

@Flux.layer EncDec

function EncDec(template_env::Environments.AbstractEnv, params::NamedTuple)
    #TODO: This shouldn't be a constructor but a factory util somewhere
    template_state = ComponentTensor(Environments.state(template_env))
    action_size = length(Environments.null_action(template_env))

    full_state_size = length(template_state)
    encoder_input_size = length(template_state.imitation_target)
    decoder_input_size = length(template_state.proprioception) + params.network.latent_dimension

    encoder_sizes = [encoder_input_size; params.network.encoder_size]   
    decoder_sizes = [decoder_input_size; params.network.decoder_size]

    critic_sizes = [full_state_size; params.network.critic_size]
                    
    return EncDec(encoder_sizes, decoder_sizes, critic_sizes, action_size;
                  bottleneck_type=params.network.bottleneck,
                  decoder_type=params.network.decoder_type,
                  sigma_min=params.network.sigma_min, sigma_max=params.network.sigma_max,
                  latent_dimension=params.network.latent_dimension, gamma=params.training.gamma,
                  loss_weight_kl=params.training.loss_weight_kl, noise_scale=params.network.noise_scale)
end

function EncDec(encoder_sizes::Vector{Int}, decoder_sizes::Vector{Int},
                critic_sizes::Vector{Int}, action_size::Integer;
                bottleneck_type=:variational, decoder_type=:MLP,
                sigma_min=0.01, sigma_max=0.5, latent_dimension=60, gamma=nothing,
                loss_weight_kl=nothing, noise_scale=nothing)
    if bottleneck_type == :variational
        bottleneck = VariationalBottleneck(encoder_sizes[end] => latent_dimension, loss_weight_kl)
    elseif bottleneck_type == :deterministic
        bottleneck = Dense(encoder_sizes[end] => latent_dimension, tanh, init=zeros32)
    elseif bottleneck_type == :informational
        bottleneck = InfoBottleneck(encoder_sizes[end] => latent_dimension, noise_scale=noise_scale, init=zeros32)
    else
        error("Unknown bottleneck type: $(bottleneck_type)")
    end

    encoder = Chain((create_layers(Dense, "encoder", encoder_sizes, tanh)..., bottleneck=bottleneck)...)
    
    if decoder_type == :MLP
        decoder = Chain((create_layers(Dense, "decoder", decoder_sizes, tanh)...,
                            mu_and_sigma=Dense(decoder_sizes[end] => 2*action_size, tanh, init=zeros32))...)
    elseif decoder_type == :LSTM
        #Q: Should the final layer actually be a Dense layer?
        decoder = Chain((create_layers(LSTM, "decoder", decoder_sizes)...,
                            mu_and_sigma=LSTM(decoder_sizes[end] => 2*action_size))...)
    end
    action_sampler = GaussianActionSampler(;sigma_min=sigma_min, sigma_max=sigma_max)

    critic  = Chain((create_layers(Dense, "critic", critic_sizes, tanh)...,
                    V_unscaled = Dense(critic_sizes[end] => 1, init=zeros32),
                    V_scaled = V_unscaled -> V_unscaled ./ eltype(V_unscaled)(1.0-gamma))...)
                    
    return EncDec(encoder, decoder, critic, action_sampler)
end

function actor(actor_critic::AbstractEncDec, state, reset_mask, action=nothing)
    #Annoying work-around to avoid auto-diff errors
    imitation_target = Flux.ignore(()->state.imitation_target |> array |> copy)
    proprioception = Flux.ignore(()->state.proprioception |> array |> copy)

    #Encoder
    latent = rollout!(actor_critic.encoder, imitation_target, reset_mask)

    #Decoder
    decoder_input = cat(latent, proprioception; dims=1)
    decoder_output = rollout!(actor_critic.decoder, decoder_input, reset_mask)

    action_and_loglikelihood = actor_critic.action_sampler(decoder_output, action)

    return action_and_loglikelihood
end

function critic(actor_critic::AbstractEncDec, state)
    input = data(state)
    return view(actor_critic.critic(input), 1, :, :)
end

function checkpoint_latent_state(actor_critic::AbstractEncDec)
    #TODO: what if the critic is stateful? Should account for that in the PPO algo.
    return (;encoder = checkpoint_latent_state(actor_critic.encoder),
             decoder = checkpoint_latent_state(actor_critic.decoder))
end

function restore_latent_state!(actor_critic::AbstractEncDec, latent_state::NamedTuple)
    restore_latent_state!(actor_critic.encoder, latent_state.encoder)
    restore_latent_state!(actor_critic.decoder, latent_state.decoder)
end

function regularization_loss(actor_critic::AbstractEncDec)
    return regularization_loss(actor_critic.encoder) +
           regularization_loss(actor_critic.decoder) +
           regularization_loss(actor_critic.critic)
end
