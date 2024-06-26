struct ActorCritic{A,C}
    actor::A
    critic::C
end

function ActorCritic(env::MuJoCoEnv, params::NamedTuple)
    state_size = length(state(env, params))
    action_size =  mapreduce(s->length(s), +, null_action(env, params))
    actor_bias = [zeros32(action_size); params.actor_sigma_init_bias*ones32(action_size)]
    actor_net = Chain(Dense(state_size => params.hidden1_size, tanh),
                      Dense(params.hidden1_size => params.hidden2_size, tanh),
                      Dense(params.hidden2_size => 2*action_size, tanh;
                            init=zeros32, bias=actor_bias))
    critic_net = Chain(Dense(state_size => params.hidden1_size, tanh),
                       Dense(params.hidden1_size => params.hidden2_size, tanh),
                       Dense(params.hidden2_size => 1; init=zeros32))
    return ActorCritic(actor_net, critic_net)
end

Flux.@layer ActorCritic

#function flatten_state(state)
    #torso_height = reshape(state.torso_height, 1, size(state.torso_height))
#    batch_dims = size(state.qpos)[2:end]
#    cat(state.qpos, state.qvel, state.act,
#        state.head_accel*0.1f0, state.head_vel, state.head_gyro,
#        state.paw_contacts, state.torso_linvel, state.torso_xmat,
#        reshape(state.torso_height, 1, size(state.torso_height)...)*10.0f0,
#        reshape(state.com_target_array, :, batch_dims...)*50.0f0; dims=1)
#end

function actor(actor_critic::ActorCritic, state, params, action=nothing)
    input = data(state) #Just flatten it
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
    input = data(state) #Just flatten it
    return view(actor_critic.critic(input), 1, :, :) ./ (1.0-params.gamma)
end

action_size(actor_critic::ActorCritic) = size(actor_critic.actor[end].weight, 1) ÷ 2
