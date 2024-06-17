struct ActorCritic{A,C}
    actor::A
    critic::C
end

function ActorCritic(env::MuJoCoEnv, params::NamedTuple)
    state_size = mapreduce(s->s isa Rectangle ? length(s.a) : 1, +, state_space(env))
    action_size = mapreduce(s->s isa Rectangle ? length(s.a) : 1, +, action_space(env))
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

function flatten_state(state)
    #torso_height = reshape(state.torso_height, 1, size(state.torso_height))
    cat(state.qpos, state.qvel, state.act,
        state.head_accel*0.1f0, state.head_vel, state.head_gyro,
        state.paw_contacts, state.torso_linvel, state.torso_xmat,
        state.torso_height*10.0f0; dims=1)
end

function actor(actor_critic::ActorCritic, state, params, action=nothing)
    input = flatten_state(state)
    actor_net_output = actor_critic.actor(input)
    action_size = size(actor_net_output, 1) ÷ 2
    batch_dims = ntuple(_->:, ndims(actor_net_output)-1)
    mu = view(actor_net_output, 1:action_size, batch_dims...)
    unscaled_sigma = view(actor_net_output, (action_size+1):2*action_size, batch_dims...)
    sigma = params.sigma_min .+ 0.5f0.*(params.sigma_max .- params.sigma_min).*(1 .+ unscaled_sigma)
    if isnothing(action)
        action = (;ctrl=mu .+ sigma .* CUDA.randn(size(mu)...))
    end
    loglikelihood = view(-0.5f0 .* sum(((action.ctrl .- mu) ./ sigma).^2; dims=1) .- sum(log.(sigma); dims=1), 1, batch_dims...)
    return (;action, loglikelihood, mu, sigma)
end

function critic(actor_critic::ActorCritic, state, params)
    input = flatten_state(state)
    return view(actor_critic.critic(input), 1, :, :) .* 1.0f3
end

action_size(actor_critic::ActorCritic) = size(actor_critic.actor[end].weight, 1) ÷ 2

#function save_checkpoint(model::ActorCritic, filename)
    #actor_state = Flux.state(model.actor)
    #critic_state = Flux.state(model.critic)
    #JLD2.jldsave(filename; actor_state, critic_state)
#end