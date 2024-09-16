struct ActorCritic{A,C,E,X,J,Ap}
    actor::A
    critic::C
    com_encoder::E
    xmat_encoder::X
    joint_encoder::J
    appendages_encoder::Ap
end

Flux.@layer ActorCritic

prop_keys() = [:qpos, :qvel, :act, :head_accel, :head_vel,
               :head_gyro,:paw_contacts, :torso_linvel,
               :torso_xmat, :torso_height]

function ActorCritic(env::MuJoCoEnv, params::NamedTuple)
    s = state(env, params)
    state_size = length(s)
    action_size =  mapreduce(s->length(s), +, null_action(env, params))
    prop_size = length(computeRange(s, prop_keys()))
    com_size = length(computeRange(s, [:com_target_array]))
    xmat_size = length(computeRange(s, [:xquat_target_array]))
    joint_size = length(computeRange(s, [:joint_target_array]))
    appendages_size = length(computeRange(s, [:appendages_target_array]))
    actor_bias = [zeros32(action_size); params.actor_sigma_init_bias*ones32(action_size)]
    actor_net = Chain(Dense((prop_size+params.latent_dimension*4) => params.hidden1_size, tanh),
                      Dense(params.hidden1_size => params.hidden2_size, tanh),
                      Dense(params.hidden2_size => 2*action_size, tanh;
                            init=zeros32, bias=actor_bias))
    critic_net = Chain(Dense(state_size => params.hidden1_size, tanh),
                       Dense(params.hidden1_size => params.hidden2_size, tanh),
                       Dense(params.hidden2_size => 1; init=zeros32))
    com_encoder = Dense(com_size=> params.latent_dimension, tanh)
    xmat_encoder = Dense(xmat_size => params.latent_dimension, tanh)
    joint_encoder = Dense(joint_size => params.latent_dimension, tanh)
    appendages_encoder = Dense(appendages_size => params.latent_dimension, tanh)
    return ActorCritic(actor_net, critic_net, com_encoder, xmat_encoder, joint_encoder, appendages_encoder)
end

function encoder(actor_critic::ActorCritic, state)
    com_target_array = Flux.ignore(()->state[:com_target_array])
    xquat_target_array = Flux.ignore(()->state[:xquat_target_array])
    joint_target_array = Flux.ignore(()->state[:joint_target_array])
    appendages_target_array = Flux.ignore(()->state[:appendages_target_array])
    prop = Flux.ignore(()->view(state, prop_keys()))
    com_encoded   = actor_critic.com_encoder(com_target_array)
    xquat_encoded = actor_critic.xmat_encoder(xquat_target_array)
    joint_encoded = actor_critic.joint_encoder(joint_target_array)
    appendages_encoded = actor_critic.appendages_encoder(appendages_target_array)
    cat(prop, com_encoded, xquat_encoded, joint_encoded, appendages_encoded; dims=1)
end

function actor(actor_critic::ActorCritic, state, params, action=nothing)
    input = encoder(actor_critic, state)
    actor_net_output = actor_critic.actor(input)
    action_size = size(actor_net_output, 1) ÷ 2
    batch_dims = ntuple(_->:, ndims(actor_net_output)-1)
    mu = view(actor_net_output, 1:action_size, batch_dims...)
    unscaled_sigma = view(actor_net_output, (action_size+1):2*action_size, batch_dims...)
    sigma = params.sigma_min .+ 0.5f0.*(params.sigma_max .- params.sigma_min).*(1 .+ unscaled_sigma)
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
    return view(actor_critic.critic(input), 1, :, :) ./ (1.0-params.gamma)
end

action_size(actor_critic::ActorCritic) = size(actor_critic.actor[end].weight, 1) ÷ 2
