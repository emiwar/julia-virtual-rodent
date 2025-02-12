struct VariationalEncDecLSTM{E,D,C}
    encoder::E
    decoder::D
    critic::C
end

Flux.@layer VariationalEncDecLSTM

function VariationalEncDecLSTM(template_env::MuJoCoEnv, params::NamedTuple)
    @warn "LSTM networks are still under development and not yet validated."
    template_state = ComponentTensor(state(template_env, params))
    action_size = length(null_action(template_env, params))

    full_state_size = length(template_state)
    encoder_input_size = length(template_state.imitation_target)
    decoder_input_size = length(template_state.proprioception) + params.network.latent_dimension

    encoder_size = params.network.encoder_size
    encoder = Chain(Dense(encoder_input_size => encoder_size[1], tanh),
                    (Dense(a => b, tanh) for (a, b) in zip(encoder_size[1:end-1], encoder_size[2:end]))...,
                    Dense(encoder_size[end] => 2*params.network.latent_dimension, tanh, init=zeros32))
    decoder_size = params.network.decoder_size
    #TODO: Is this one LSTM layer too many? 
    decoder = Chain(LSTM(decoder_input_size => decoder_size[1]),
                    (LSTM(a => b) for (a, b) in zip(decoder_size[1:end-1], decoder_size[2:end]))...,
                    LSTM(decoder_size[end] => 2*action_size))
    critic_size = params.network.critic_size
    critic  = Chain(Dense(full_state_size => critic_size[1], tanh),
                    (Dense(a => b, tanh) for (a, b) in zip(critic_size[1:end-1], critic_size[2:end]))...,
                    Dense(critic_size[end] => 1, init=zeros32))

    return VariationalEncDecLSTM(encoder, decoder, critic)
end

has_latent_layer(::VariationalEncDecLSTM) = true

randn_like(arr::AbstractArray) = randn(size(arr)...)
randn_like(arr::CUDA.AnyCuArray) = CUDA.randn(size(arr)...)

function actor(actor_critic::VariationalEncDecLSTM, state, inv_reset_mask, params, action=nothing, latent_eps=nothing)
    #Annoying work-around to avoid auto-diff errors
    imitation_target = Flux.ignore(()->state.imitation_target |> array |> copy)
    proprioception = Flux.ignore(()->state.proprioception |> array |> copy)
    batch_dims = ntuple(_->:, ndims(proprioception)-1)

    #Encoder
    latent_dimension = params.network.latent_dimension
    encoder_output = actor_critic.encoder(imitation_target)
    latent_mu = @view encoder_output[1:latent_dimension, batch_dims...]
    latent_logsigma = 3.0f0 .* (@view encoder_output[latent_dimension+1:end, batch_dims...])
    if isnothing(latent_eps)
        latent_eps = randn_like(latent_mu)
    end
    latent = latent_mu .+ exp.(latent_logsigma) .* latent_eps

    decoder_input = cat(latent, proprioception; dims=1)
    decoder_output = cat(map(axes(decoder_input, 3)) do t #This should work also for ndims=2
        for layer in actor_critic.decoder
            layer.state = map(s->s .* view(inv_reset_mask, :, t)', layer.state)
        end
        actor_critic.decoder(view(decoder_input, :, :, t))
    end...; dims=3)

    #Draw action with mean and sigma, and compute action likelihood
    action_size = size(decoder_output, 1) ÷ 2
    mu = @view decoder_output[1:action_size, batch_dims...]
    unscaled_sigma = @view decoder_output[action_size+1:end, batch_dims...]
    sigma_min = params.network.sigma_min
    sigma_max = params.network.sigma_max
    sigma = sigma_min .+ 0.5f0.*(sigma_max .- sigma_min).*(1 .+ unscaled_sigma)
    if isnothing(action)
        xsi = randn_like(mu)
        action = mu .+ sigma .* xsi
    end
    loglikelihood = -0.5f0 .* sum(((action .- mu) ./ sigma).^2; dims=1) .- sum(log.(sigma); dims=1)
    (;action, mu, sigma, loglikelihood, latent, latent_mu, latent_logsigma, latent_eps)
end

function critic(actor_critic::VariationalEncDecLSTM, state, params)
    input = data(state)
    return view(actor_critic.critic(input), 1, :, :) ./ (1.0-params.training.gamma)
end

decoder_is_stateful(actor_critic) = any(isa.(actor_critic.decoder, Flux.Recur))

function checkpoint_state(actor_critic::VariationalEncDecLSTM)
    map(layer->layer.state, actor_critic.decoder)
end

function restore_state!(actor_critic::VariationalEncDecLSTM, hidden)
    for (layer, h) in zip(actor_critic.decoder, hidden)
        layer.state = h
    end
end

function reset_to_dim!(actor_critic::VariationalEncDecLSTM, batch_size)
    decoder_input_size = size(actor_critic.decoder[1].cell.Wi, 2)
    actor_critic.decoder(CUDA.zeros(decoder_input_size, batch_size...))
    #Flux.reset!(actor_critic)
end

function decoder_only(actor_critic::VariationalEncDecLSTM, state, latent, params; action_noise=false)
    error("Not implemented for VariationalEncDecLSTM")
end

#action_size(actor_critic::ActorCritic) = size(actor_critic.actor[end].weight, 1) ÷ 2

function Base.isapprox(a::Vector{Tuple{T, T}}, b::Vector{Tuple{T, T}}) where {T}
    if length(a) != length(b) 
        return false
    end
    for (a_, b_) in zip(a, b)
        if !(a_[1] ≈ b_[1] && a_[2] ≈ b_[2])
            return false
        end
    end
    return true
end

