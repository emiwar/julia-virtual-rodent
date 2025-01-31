abstract type Collector end

struct CuCollector{S, I, A} <: Collector
    states::S
    rewards::CUDA.CuMatrix{Float32}
    status::CUDA.CuMatrix{UInt8}
    infos::I
    actor_outputs::A
end

function CuCollector(template_env, template_actor, n_envs, steps_per_batch; actor_fcn=actor)
    template_state  = state(template_env, params) |> ComponentTensor
    template_info   = info(template_env, params) |> ComponentTensor
    template_actor_output = actor_fcn(template_actor, template_state, params) |> ComponentTensor
    #template_status = fill(RUNNING, 1)
    #template_actor_output = actor_fcn(template_actor, template_state, template_status, params) |> ComponentTensor
    
    #Big GPU arrays/BatchComponentTensor for storing the entire batch
    states = BatchComponentTensor(template_state, n_envs, steps_per_batch+1; array_fcn=CUDA.zeros)
    rewards = CUDA.zeros(n_envs, steps_per_batch)
    status = CUDA.zeros(UInt8, n_envs, steps_per_batch+1)
    actor_outputs =  BatchComponentTensor(template_actor_output, n_envs, steps_per_batch; array_fcn=CUDA.zeros)

    #...except infos, which never have to be moved to the GPU
    infos = BatchComponentTensor(template_info, n_envs, steps_per_batch; array_fcn=zeros)
    
    CuCollector(states, rewards, status, infos, actor_outputs)
end

function collect_batch!(actor::Function, collector::Collector, stepper,
                        params, lapTimer; action_preprocess::Function=(a,s,p)->a)
    steps_per_batch = params.rollout.n_steps_per_epoch
    lap(lapTimer, :first_state)
    prepareEpoch!(stepper, params)
    collector.states[:, :, 1] = stepper.states
    collector.status[:, 1] = stepper.status
    for t=1:steps_per_batch
        lap(lapTimer, :rollout_actor)
        #collector.actor_outputs[:, :, t] = actor((@view collector.states[:, :, t]), (@view collector.status[:, t]), params)
        collector.actor_outputs[:, :, t] = actor((@view collector.states[:, :, t]), params)
        actions = action_preprocess((@view collector.actor_outputs[:action, :, t]),
                                    (@view collector.states[:, :, t]), params)
        lap(lapTimer, :rollout_action_to_cpu)
        copyto!(stepper.actions, actions)
        step!(stepper, params, lapTimer)
        collector.states[:, :, t+1] = stepper.states
        collector.infos[:, :, t] = stepper.infos
        collector.rewards[:, t] = stepper.rewards
        collector.status[:, t+1] = stepper.status
    end
end

function collect_batch!(stepper::MpiStepperWorker, params, lapTimer)
    steps_per_batch = params.rollout.n_steps_per_epoch
    prepareEpoch!(stepper, params)
    for t=1:steps_per_batch
        step!(stepper, params, lapTimer)
    end
end

function compute_batch_stats(collector::Collector)
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