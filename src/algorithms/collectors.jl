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
    template_state  = state(env, params) |> ComponentTensor
    template_info   = info(env, params) |> ComponentTensor

    n_envs = size(template_state, 2)
    template_reset_mask = fill(false, n_envs)
    template_actor_output = actor(template_actor, template_state, template_reset_mask) |> ComponentTensor

    #Big GPU arrays/BatchComponentTensor for storing the entire batch
    states = BatchComponentTensor(template_state, steps_per_batch+1; array_fcn=CUDA.zeros)
    rewards = CUDA.zeros(n_envs, steps_per_batch)
    status = CUDA.zeros(UInt8, n_envs, steps_per_batch+1)
    actor_outputs =  BatchComponentTensor(template_actor_output, steps_per_batch; array_fcn=CUDA.zeros)

    #...except infos, which never have to be moved to the GPU
    infos = BatchComponentTensor(template_info, steps_per_batch+1; array_fcn=zeros)
    
    CuCollector(env, states, rewards, status, infos, actor_outputs)
end

function collect_epoch!(collector::Collector, networks)
    lap(:first_state)
    prepare_epoch!(collector.env)
    collector.states[:, :, 1] = state(collector.env)
    collector.status[:, 1] = status(collector.env)
    collector.infos[:, :, 1] = info(collector.env)
    for t=1:steps_per_batch(collector)
        lap(:rollout_actor)

        #Some networks are stateful and need to be reset when the environment is reset.
        reset_mask = (@view collector.status[:, t]) .!= RUNNING
        prev_state = @view collector.states[:, :, t]

        #Run the actor (on the GPU)
        collector.actor_outputs[:, :, t] = actor(networks, prev_state, reset_mask)

        #Apply the action and step the environment
        act!(collector.env, actions)

        #Record the transition
        lap(:rollout_move_to_gpu)
        collector.states[:, :, t+1] = state(env)
        collector.infos[:, :, t+1] = info(env)
        collector.rewards[:, t] = reward(env)
        collector.status[:, t+1] = status(env)
    end
end

n_envs(collector::CuCollector) = size(collector.rewards, 1)
steps_per_batch(collector::CuCollector) = size(collector.rewards, 2)

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
    logdict["rollout_batch/termination_rate"] = sum(collector.status .== TERMINATED) / length(collector.status)
    batch_done = Array(collector.status .!= RUNNING)
    merge!(logdict, quantile_dict("rollout_batch/lifespan", view(array(collector.infos.lifetime), 1, :, :)[batch_done]))
    for key in keys(index(collector.infos))
        merge!(logdict, quantile_dict("rollout_batch/$key", collector.infos[key]))
    end
    return logdict
end