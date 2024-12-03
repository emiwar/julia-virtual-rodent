struct JoystickMLP{A,C}
    actor::A
    critic::C
end

Flux.@layer JoystickMLP

function JoystickMLP(template_env::RodentJoystickEnv, params::NamedTuple)
    template_state = ComponentTensor(state(template_env, params))
    input_size = length(template_state.command)
    action_size = params.network.latent_dimension
    full_state_size = length(template_state)

    actor_size = params.network.actor_size
    actor = Chain(Dense(input_size => actor_size[1], tanh),
                  (Dense(a => b, tanh) for (a, b) in zip(actor_size[1:end-1], actor_size[2:end]))...,
                  Dense(actor_size[end] => 2*action_size, tanh, init=zeros32))
    critic_size = params.network.critic_size
    critic  = Chain(Dense(full_state_size => critic_size[1], tanh),
                    (Dense(a => b, tanh) for (a, b) in zip(critic_size[1:end-1], critic_size[2:end]))...,
                    Dense(critic_size[end] => 1, init=zeros32))

    return JoystickMLP(actor, critic)
end

randn_like(arr::AbstractArray) = randn(size(arr)...)
randn_like(arr::CUDA.AnyCuArray) = CUDA.randn(size(arr)...)

function actor(joystickMLP::JoystickMLP, state, params, action=nothing)
    #Annoying work-around to avoid auto-diff errors
    command = Flux.ignore(()->state.command |> array |> copy)
    batch_dims = ntuple(_->:, ndims(command)-1)

    #Encoder
    control_output = joystickMLP.actor(command)

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

function critic(joystickMLP::JoystickMLP, state, params)
    input = data(state)
    return view(joystickMLP.critic(input), 1, :, :) ./ (1.0-params.training.gamma)
end
