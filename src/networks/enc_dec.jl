abstract type AbstractEncDec end

struct EncDec{E <: Chain, D <: Chain, C <: Chain, A <: ActionSampler} <: AbstractEncDec
    encoder::E
    decoder::D
    action_sampler::A
    critic::C
end

function EncDec(template_env::MuJoCoEnv, params::NamedTuple)
    template_state = ComponentTensor(state(template_env, params))
    action_size = length(null_action(template_env, params))

    full_state_size = length(template_state)
    encoder_input_size = length(template_state.imitation_target)
    decoder_input_size = length(template_state.proprioception) + params.network.latent_dimension

    encoder_sizes = [encoder_input_size; params.network.encoder_size]
    if params.networks.bottleneck == :variational
        bottleneck = VariationalBottleneck(encoder_sizes[end] => params.network.latent_dimension)
    elseif params.networks.bottleneck == :deterministic
        bottleneck = Dense(encoder_sizes[end] => params.network.latent_dimension, tanh, init=zeros32)
    end

    encoder = Chain(create_layers(Dense, "encoder", encoder_sizes, tanh)...; bottleneck)
    
    decoder_sizes = [decoder_input_size; params.network.decoder_size]

    if params.networks.decoder_type == :MLP
        decoder = Chain(create_layers(Dense, "decoder", decoder_sizes, tanh)...,
                        Dense(decoder_sizes[end] => 2*action_size, tanh, init=zeros32))
    elseif params.networks.decoder_type == :LSTM
        #Q: Should the final layer actually be a Dense layer?
        decoder = Chain(create_layers(LSTM, "decoder", decoder_sizes, tanh)...,
                        LSTM(decoder_sizes[end] => 2*action_size, tanh, init=zeros32))
    end
    action_sampler = GaussianActionSampler(;sigma_min=params.network.sigma_min,
                                            sigma_max=params.network.sigma_max)

    critic_sizes = [full_state_size; params.network.critic_size]
    critic  = Chain(create_layers(Dense, "critic", critic_sizes, tanh)...,
                    Dense(critic_sizes[end] => 1, init=zeros32),
                    V(critic_ouput) = critic_output ./ eltype(critic_output)(1.0-params.training.gamma))
                    
    return EncDec(encoder, decoder, critic, action_sampler)
end

function actor(actor_critic::AbstractEncDec, state, reset_mask, action=nothing)
    #Annoying work-around to avoid auto-diff errors
    imitation_target = Flux.ignore(()->state.imitation_target |> array |> copy)
    proprioception = Flux.ignore(()->state.proprioception |> array |> copy)

    #Encoder
    latent = rollout(actor_critic.encoder, imitation_target, reset_mask)

    #Decoder
    decoder_input = cat(latent, proprioception; dims=1)
    decoder_output = rollout(actor_critic.decoder, decoder_input, reset_mask)

    total_regularization_loss = regularization_loss(actor_critic.encoder) + regularization_loss(actor_critic.decoder)

    action_and_loglikelihood = actor_critic.action_sampler(decoder_output, action)

    return (;action_and_loglikelihood..., total_regularization_loss)
end

function critic(actor_critic::AbstractEncDec, state)
    input = data(state)
    return view(actor_critic.critic(input), 1, :, :)
end

function checkpoint_latent_state(actor_critic::AbstractEncDec)
    #TODO: what if the critic is stateful? Should account for that in PPO loss.
    return (;encoder = checkpoint_latent_state(actor_critic.encoder),
             decoder = checkpoint_latent_state(actor_critic.decoder))
end

function restore_latent_state!(actor_critic::AbstractEncDec, latent_state)
    restore_latent_state!(actor_critic.encoder, latent_state.encoder)
    restore_latent_state!(actor_critic.decoder, latent_state.decoder)
end
