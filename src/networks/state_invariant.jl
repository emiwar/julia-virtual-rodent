struct StateInvariantActor{A,C}
    actor::A
    critic::C
end
Flux.@layer StateInvariantActor

function StateInvariantActor(template_env, params::NamedTuple)
    template_state = ComponentTensor(state(template_env, params))
    full_state_size = length(template_state)
    action_size = params.network.latent_dimension
    actor = Dense(1=>2*action_size)
    
    critic_size = params.network.critic_size
    critic  = Chain(Dense(full_state_size => critic_size[1], tanh),
                    (Dense(a => b, tanh) for (a, b) in zip(critic_size[1:end-1], critic_size[2:end]))...,
                    Dense(critic_size[end] => 1, init=zeros32))

    return StateInvariantActor(actor, critic)
end
has_latent_layer(::StateInvariantActor) = false

randn_like(arr::AbstractArray) = randn(size(arr)...)
randn_like(arr::CUDA.AnyCuArray) = CUDA.randn(size(arr)...)
onedim_arr(arr::AbstractArray) = zeros(Float32, 1, size(arr)[2:end]...)
onedim_arr(arr::CUDA.AnyCuArray) = CUDA.zeros(Float32, 1, size(arr)[2:end]...)

function actor(actor_critic::StateInvariantActor, state, params, action=nothing)
    #Annoying work-around to avoid auto-diff errors
    batch_dims = ntuple(_->:, ndims(state)-1)

    #Encoder
    dummy_input = Flux.ignore(()->onedim_arr(array(state)))
    control_output = actor_critic.actor(dummy_input)

    #Draw action with mean and sigma, and compute action likelihood
    action_size = size(control_output, 1) ÷ 2
    mu = @view control_output[1:action_size, batch_dims...]
    unscaled_sigma = @view control_output[action_size+1:end, batch_dims...]
    sigma_min = params.network.sigma_min
    sigma_max = params.network.sigma_max
    sigma = sigma_min .+ 0.5f0.*(sigma_max .- sigma_min).*(1 .+ unscaled_sigma)
    if isnothing(action)
        xsi = randn_like(mu)
        action = mu .+ sigma .* xsi
    end
    loglikelihood = -0.5f0 .* sum(((action .- mu) ./ sigma).^2; dims=1) .- sum(log.(sigma); dims=1)
    (;action, mu, sigma, loglikelihood)
end

function critic(actor_critic::StateInvariantActor, state, params)
    input = data(state)
    return view(actor_critic.critic(input), 1, :, :) ./ (1.0-params.training.gamma)
end
