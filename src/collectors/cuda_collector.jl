abstract type Collector end

struct CuCollector{S, I, A} <: Collector
    states::S
    rewards::CUDA.CuMatrix{Float32}
    status::CUDA.CuMatrix{UInt8}
    infos::I
    actor_outputs::A
end

function CuCollector(template_env, template_actor, n_envs, steps_per_batch)
    template_state  = state(template_env, params) |> ComponentTensor
    template_info   = info(template_env, params) |> ComponentTensor
    #template_actor_output = actor_fcn(template_actor, template_state, params) |> ComponentTensor
    template_reset_mask = false#fill(false, 1)
    template_actor_output = actor(template_actor, template_state, template_reset_mask) |> ComponentTensor
    #Big GPU arrays/BatchComponentTensor for storing the entire batch
    states = BatchComponentTensor(template_state, n_envs, steps_per_batch+1; array_fcn=CUDA.zeros)
    rewards = CUDA.zeros(n_envs, steps_per_batch)
    status = CUDA.zeros(UInt8, n_envs, steps_per_batch+1)
    actor_outputs =  BatchComponentTensor(template_actor_output, n_envs, steps_per_batch; array_fcn=CUDA.zeros)

    #...except infos, which never have to be moved to the GPU
    infos = BatchComponentTensor(template_info, n_envs, steps_per_batch+1; array_fcn=zeros)
    
    CuCollector(states, rewards, status, infos, actor_outputs)
end

function collect_batch!(collector::Collector, networks, stepper, params, lapTimer)
    steps_per_batch = params.rollout.n_steps_per_epoch
    lap(lapTimer, :first_state)
    prepareEpoch!(stepper, params)
    collector.states[:, :, 1] = stepper.states
    collector.status[:, 1] = stepper.status
    collector.infos[:, :, 1] = stepper.infos
    for t=1:steps_per_batch
        lap(lapTimer, :rollout_actor)

        #Some networks are stateful and need to be reset when the environment is reset.
        reset_mask = (@view collector.status[:, t]) .!= RUNNING
        #Run the actor on the GPU
        collector.actor_outputs[:, :, t] = actor(networks, (@view collector.states[:, :, t]), reset_mask)
        
        #Some environments (like the joystick) need to do some calculations on the
        #GPU (running a pretrained decoder) before the actions are moved to the CPU.
        #This is a workaround for that.
        lap(lapTimer, :preprocess_actions)
        actions = preprocess_actions(env_type(stepper),
                                     (@view collector.actor_outputs[:action, :, t]),
                                     (@view collector.states[:, :, t]), params)

        #Move the actions to the CPU and run the stepper (multithreaded or MPI)
        lap(lapTimer, :rollout_action_to_cpu)
        copyto!(stepper.actions, actions)
        step!(stepper, params, lapTimer)
        collector.states[:, :, t+1] = stepper.states
        collector.infos[:, :, t+1] = stepper.infos
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