abstract type Collector end

struct CuCollector{E, S, I, A} <: Collector
    env::E
    states::S
    rewards::CUDA.CuMatrix{Float32}
    status::CUDA.CuMatrix{UInt8}
    infos::I
    actor_outputs::A
end

function CuCollector(env, template_actor, steps_per_batch)
    template_state  = Environments.state(env)
    template_info   = Environments.info(env)
    state_size, n_envs = size(template_state)
    info_size, n_envs = size(template_info)

    template_actor_output = Networks.actor(template_actor, (@view template_state[:, 1]),
                                           false) |> ComponentArray
    actor_output_size, = size(template_actor_output)
    
    #Big GPU ComponentArrays for storing the entire batch
    states = ComponentArray(CUDA.zeros(state_size, n_envs, steps_per_batch), (getaxes(template_state)[1], FlatAxis(), FlatAxis()))
    rewards = CUDA.zeros(n_envs, steps_per_batch)
    status = CUDA.zeros(UInt8, n_envs, steps_per_batch+1)
    actor_outputs = ComponentArray(CUDA.zeros(actor_output_size, n_envs, steps_per_batch), (getaxes(template_actor_output)[1], FlatAxis(), FlatAxis()))

    #...except infos, which never have to be moved to the GPU
    infos = ComponentArray(zeros(state_size, n_envs, steps_per_batch), (getaxes(template_info)[1], FlatAxis(), FlatAxis()))
    
    CuCollector(env, states, rewards, status, infos, actor_outputs)
end

function collect_epoch!(collector::Collector, networks)
    lap(:first_state)
    Environments.prepare_epoch!(collector.env)
    collector.states[:, :, 1] = Environments.state(collector.env)
    collector.status[:, 1] = Environments.status(collector.env)
    collector.infos[:, :, 1] = Environments.info(collector.env)
    for t=1:steps_per_epoch(collector)
        lap(:rollout_actor)

        #Some networks are stateful and need to be reset when the environment is reset.
        reset_mask = (@view collector.status[:, t]) .!= Environments.RUNNING
        prev_state = @view collector.states[:, :, t]

        #Run the actor (on the GPU)
        collector.actor_outputs[:, :, t] = Networks.actor(networks, prev_state, reset_mask)
        actions = @view collector.actor_outputs[:action, :, t]

        #Apply the action and step the environment
        Environments.act!(collector.env, actions)

        #Record the transition
        lap(:rollout_move_to_gpu)
        collector.states[:, :, t+1] = Environments.state(collector.env)
        collector.infos[:, :, t+1] = Environments.info(collector.env)
        collector.rewards[:, t] = Environments.reward(collector.env)
        collector.status[:, t+1] = Environments.status(collector.env)
    end
end

n_envs(collector::CuCollector) = size(collector.rewards, 1)
steps_per_epoch(collector::CuCollector) = size(collector.rewards, 2)


function compute_batch_stats(collector::Collector)
    lap(:logging_batch_stats)
    logdict = Dict{String, Number}()
    merge!(logdict, quantile_dict("actor/mus", view( collector.actor_outputs, :mu, :, :)))
    merge!(logdict, quantile_dict("actor/sigmas", view( collector.actor_outputs, :sigma, :, :)))
    merge!(logdict, quantile_dict("actor/action_ctrl", view( collector.actor_outputs, :action, :, :)))
    merge!(logdict, quantile_dict("actor/action_ctrl_sum_squared", sum(view(collector.actor_outputs, :action, :, :).^2; dims=1)))
    if :latent in keys(collector.actor_outputs)
        merge!(logdict, quantile_dict("actor/latent", view(collector.actor_outputs, :latent, :, :)))
    end
    if :latent_mu in keys(collector.actor_outputs)
        merge!(logdict, quantile_dict("actor/latent_mu", view(collector.actor_outputs, :latent_mu, :, :)))
    end
    if :latent_logsigma in keys(collector.actor_outputs)
        merge!(logdict, quantile_dict("actor/latent_logsigma", view(collector.actor_outputs, :latent_logsigma, :, :)))
    end
    merge!(logdict, quantile_dict("rollout_batch/rewards", collector.rewards))
    logdict["rollout_batch/termination_rate"] = sum(collector.status .== Environments.TERMINATED) / length(collector.status)
    batch_done = Array(collector.status .!= Environments.RUNNING)
    merge!(logdict, quantile_dict("rollout_batch/lifespan", view(array(collector.infos.lifetime), 1, :, :)[batch_done]))
    for key in keys(index(collector.infos))
        merge!(logdict, quantile_dict("rollout_batch/$key", collector.infos[key]))
    end
    return logdict
end