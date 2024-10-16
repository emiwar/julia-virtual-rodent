struct ActorCritic{A,C,E,X,J,Ap}
    actor::A
    critic::C
    com_encoder::E
    root_quat_encoder::X
    joint_encoder::J
    appendages_encoder::Ap
end

Flux.@layer ActorCritic

function ActorCritic(env::MuJoCoEnv, params::NamedTuple)
    s = ComponentTensor(state(env, params))
    state_size = length(s)
    action_size =  length(null_action(env, params))#mapreduce(s->length(s), +, null_action(env, params))
    #prop_size = length(computeRange(s, prop_keys()))
    #com_size = length(computeRange(s, [:com_target_array]))
    #xmat_size = length(computeRange(s, [:xquat_target_array]))
    #joint_size = length(computeRange(s, [:joint_target_array]))
    #appendages_size = length(computeRange(s, [:appendages_target_array]))
    actor_input_size = length(s.proprioception) + params.network.latent_dimension*4
    actor_bias = [zeros32(action_size); params.network.sigma_init_bias*ones32(action_size)]
    actor_net = Chain(Dense(actor_input_size => params.network.hidden1_size, tanh),
                      Dense(params.network.hidden1_size => params.network.hidden2_size, tanh),
                      Dense(params.network.hidden2_size => 2*action_size, tanh;
                            init=zeros32, bias=actor_bias))
    critic_net = Chain(Dense(state_size => params.network.hidden1_size, tanh),
                       Dense(params.network.hidden1_size => params.network.hidden2_size, tanh),
                       Dense(params.network.hidden2_size => 1; init=zeros32))
    com_encoder = Dense(length(s.imitation_target.com) => params.network.latent_dimension, tanh)
    root_quat_encoder = Dense(length(s.imitation_target.root_quat) => params.network.latent_dimension, tanh)
    joint_encoder = Dense(length(s.imitation_target.joints) => params.network.latent_dimension, tanh)
    appendages_encoder = Dense(length(s.imitation_target.appendages) => params.network.latent_dimension, tanh)
    #encoders = map(keys(s.imitation_target)) do key
    #    input_size = length(view(s.imitation_target, key))
    #    key => Dense(input_size => params.network.latent_dimension, tanh)
    #end |> NamedTuple
    return ActorCritic(actor_net, critic_net, com_encoder, root_quat_encoder, joint_encoder, appendages_encoder)
end

function encoder(actor_critic::ActorCritic, state)
    com_target_array = Flux.ignore(()->state.imitation_target.com |> array |> copy)
    root_quat_target_array = Flux.ignore(()->state.imitation_target.root_quat |> array |> copy)
    joint_target_array = Flux.ignore(()->state.imitation_target.joints |> array |> copy)
    appendages_target_array = Flux.ignore(()->state.imitation_target.appendages |> array |> copy)
    proprioception = Flux.ignore(()->state.proprioception |> array |> copy)
    #println(typeof(com_target_array), size(com_target_array))
    com_encoded   = actor_critic.com_encoder(com_target_array)
    root_quat_encoded = actor_critic.root_quat_encoder(root_quat_target_array)
    joint_encoded = actor_critic.joint_encoder(joint_target_array)
    appendages_encoded = actor_critic.appendages_encoder(appendages_target_array)
    cat(proprioception, com_encoded, root_quat_encoded, joint_encoded, appendages_encoded; dims=1)
end

function actor(actor_critic::ActorCritic, state, params, action=nothing)
    input = encoder(actor_critic, state)
    actor_net_output = actor_critic.actor(input)
    action_size = size(actor_net_output, 1) ÷ 2
    batch_dims = ntuple(_->:, ndims(actor_net_output)-1)
    mu = view(actor_net_output, 1:action_size, batch_dims...)
    unscaled_sigma = view(actor_net_output, (action_size+1):2*action_size, batch_dims...)
    sigma_min = params.network.sigma_min
    sigma_max = params.network.sigma_max
    sigma = sigma_min .+ 0.5f0.*(sigma_max .- sigma_min).*(1 .+ unscaled_sigma)
    if isnothing(action)
        if mu isa CUDA.AnyCuArray
            xsi = CUDA.randn(size(mu)...)
        else
            xsi = randn(size(mu)...)
        end
        action = mu .+ sigma .* xsi
    end
    loglikelihood = -0.5f0 .* sum(((action .- mu) ./ sigma).^2; dims=1) .- sum(log.(sigma); dims=1)
    (;action, mu, sigma, loglikelihood)
end

function critic(actor_critic::ActorCritic, state, params)
    input = data(state)
    return view(actor_critic.critic(input), 1, :, :) ./ (1.0-params.training.gamma)
end

action_size(actor_critic::ActorCritic) = size(actor_critic.actor[end].weight, 1) ÷ 2
