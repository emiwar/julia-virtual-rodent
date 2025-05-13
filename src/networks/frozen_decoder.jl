#TODO: this should maybe be in a separate module since it's an env wrapper that needs
#      a lot of `Networks` functionality
struct FrozenDecoder{D, E, PB} <: Environments.AbstractEnv
    decoder::D
    base_env::E
    proprioception_buffer::PB
end

function FrozenDecoder(decoder, base_env::Environments.AbstractEnv) 
    template_state = Environments.state(base_env)
    return FrozenDecoder(decoder, base_env, ComponentArray(template_state))
end

Environments.state(env::FrozenDecoder) = Environments.state(env.base_env)
Environments.reward(env::FrozenDecoder) = Environments.reward(env.base_env)
Environments.info(env::FrozenDecoder) = Environments.info(env.base_env)
Environments.status(env::FrozenDecoder) = Environments.status(env.base_env)

function Environments.act!(env::FrozenDecoder, actions)
    reset_mask, proprioception = Flux.ignore() do
        env.proprioception_buffer[:] = Environments.state(env).proprioception
        Environments.status(env) .!= Environments.RUNNING, env.proprioception_buffer |> Flux.gpu
    end

    #Treat actions as latent of the decoder
    decoder_input = cat(actions, proprioception; dims=1)
    decoder_output = rollout!(env.decoder, decoder_input, reset_mask)

    mu, unscaled_sigma = split_halfway(decoder_output; dim=1)

    #No sampling here, we just use the mean actions
    base_actions = mu
    
    # Apply the decoded actions to the base environment
    Environments.act!(env.base_env, base_actions)
end

