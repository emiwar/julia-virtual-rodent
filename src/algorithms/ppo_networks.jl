struct ActorCritic{E,D,C}
    encoder::E
    decoder::D
    critic::C
end

Flux.@layer ActorCritic

function ActorCritic(template_env::MuJoCoEnv, params::NamedTuple)
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
    decoder = Chain(Dense(decoder_input_size => decoder_size[1], tanh),
                    (Dense(a => b, tanh) for (a, b) in zip(decoder_size[1:end-1], decoder_size[2:end]))...,
                    Dense(decoder_size[end] => 2*action_size, tanh, init=zeros32))
    critic_size = params.network.critic_size
    critic  = Chain(Dense(full_state_size => critic_size[1], tanh),
                    (Dense(a => b, tanh) for (a, b) in zip(critic_size[1:end-1], critic_size[2:end]))...,
                    Dense(critic_size[end] => 1, init=zeros32))

    return ActorCritic(encoder, decoder, critic)
end

randn_like(arr::AbstractArray) = randn(size(arr)...)
randn_like(arr::CUDA.AnyCuArray) = CUDA.randn(size(arr)...)

function actor(actor_critic::ActorCritic, state, params, action=nothing, latent_eps=nothing)
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

    #Decoder
    decoder_input = cat(latent, proprioception; dims=1)
    decoder_output = actor_critic.decoder(decoder_input)

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

function critic(actor_critic::ActorCritic, state, params)
    input = data(state)
    return view(actor_critic.critic(input), 1, :, :) ./ (1.0-params.training.gamma)
end

function decoder_only(actor_critic::ActorCritic, state, latent, params; action_noise=false)
    proprioception = Flux.ignore(()->state.proprioception |> array |> copy)
    batch_dims = ntuple(_->:, ndims(proprioception)-1)
    decoder_input = cat(latent, proprioception; dims=1)
    decoder_output = actor_critic.decoder(decoder_input)

    #Draw action with mean and sigma, and compute action likelihood
    action_size = size(decoder_output, 1) ÷ 2
    mu = @view decoder_output[1:action_size, batch_dims...]
    if action_noise
        unscaled_sigma = @view decoder_output[action_size+1:end, batch_dims...]
        sigma_min = params.network.sigma_min
        sigma_max = params.network.sigma_max
        sigma = sigma_min .+ 0.5f0.*(sigma_max .- sigma_min).*(1 .+ unscaled_sigma)
        xsi = randn_like(mu)
        return mu .+ sigma .* xsi
    else
        return mu
    end
end

#action_size(actor_critic::ActorCritic) = size(actor_critic.actor[end].weight, 1) ÷ 2
