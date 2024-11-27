function BatchCollectorRoot(envs, actor_critic, params)#, lapTimer::LapTimer)
    action_size =  mapreduce(s->length(s), +, Main.MuJoCoEnvs.null_action(envs[1], params))
    steps_per_batch = params.rollout.n_steps_per_epoch
    n_envs = params.rollout.n_envs
    n_local_envs = length(envs)

    #Big GPU arrays/BatchComponentTensor for storing the entire batch
    template_state = Main.ComponentTensor(Main.MuJoCoEnvs.state(envs[1], params))
    states = Main.BatchComponentTensor(template_state, n_envs, steps_per_batch+1; array_fcn=CUDA.zeros)
    template_actor_output = Main.Networks.actor(actor_critic, view(states, :, 1, 1), params) |> Main.ComponentTensor
    actor_output =  Main.BatchComponentTensor(template_actor_output, n_envs, steps_per_batch; array_fcn=CUDA.zeros)
    rewards = CUDA.zeros(        n_envs, steps_per_batch)
    env_status = CUDA.zeros(UInt8, n_envs, steps_per_batch)

    #...except infos, which never have to be moved to the GPU
    template_info = Main.ComponentTensor(Main.MuJoCoEnvs.info(envs[1], params))
    infos = Main.BatchComponentTensor(template_info, n_envs, steps_per_batch; array_fcn=zeros)

    #Smaller arrays/ComponentTensor for keeping one timestep in CPU memory while multithreading
    step_states = Main.BatchComponentTensor(template_state, n_envs, array_fcn=(inds...)->zeros(Float32, inds...))
    step_reward = zeros(Float32, n_envs)
    step_status = zeros(UInt8, n_envs)
    step_info  = Main.BatchComponentTensor(template_info, n_envs, array_fcn=(inds...)->zeros(Float32, inds...))
    step_actions = zeros(Float32, action_size, n_envs)

    local_step_states = Main.BatchComponentTensor(template_state, n_local_envs, array_fcn=(inds...)->zeros(Float32, inds...))
    local_step_reward = zeros(Float32, n_local_envs)
    local_step_status = zeros(UInt8, n_local_envs)
    local_step_infos  = Main.BatchComponentTensor(template_info, n_local_envs, array_fcn=(inds...)->zeros(Float32, inds...))
    local_step_actions = zeros(Float32, action_size, n_local_envs)
    function collect(lapTimer)#::LapTimer)
        Main.lap(lapTimer, :resetting_envs)
        #States for the first timestep
        @Threads.threads for i=1:n_local_envs
            if params.rollout.reset_on_epoch_start
                Main.MuJoCoEnvs.reset!(envs[i], params)
            end
            local_step_states[:, i] = Main.MuJoCoEnvs.state(envs[i], params)
        end
        Main.lap(lapTimer, :mpi_gather_first_state)
        MPI.Gather!(local_step_states |> Main.data, step_states |> Main.data, MPI.COMM_WORLD)
        Main.lap(lapTimer, :first_state_to_gpu)
        states[:, :, 1] = step_states
        for t=1:steps_per_batch
            Main.lap(lapTimer, :rollout_actor)
            actor_output[:, :, t] = Main.Networks.actor(actor_critic, view(states, :, :, t), params)
            Main.lap(lapTimer, :rollout_action_to_cpu)
            CUDA.copyto!(step_actions, view(actor_output, :action, :, t))
            Main.lap(lapTimer, :mpi_scatter_actions)
            MPI.Scatter!(step_actions, local_step_actions, MPI.COMM_WORLD)
            Main.lap(lapTimer, :rollout_envs)
            @Threads.threads for i=1:n_local_envs
                env = envs[i]
                action = view(local_step_actions, :, i)
                Main.MuJoCoEnvs.act!(env, action, params)
                local_step_states[:, i] = Main.MuJoCoEnvs.state(envs[i], params)
                local_step_reward[i] = Main.MuJoCoEnvs.reward(env, params)
                local_step_status[i] = Main.MuJoCoEnvs.status(env, params)
                local_step_infos[:, i] = Main.MuJoCoEnvs.info(envs[i], params)
                if local_step_status[i] != Main.MuJoCoEnvs.RUNNING
                    Main.MuJoCoEnvs.reset!(env, params)
                end
            end
            Main.lap(lapTimer, :mpi_wait_for_workers)
            MPI.Barrier(MPI.COMM_WORLD)
            Main.lap(lapTimer, :mpi_gather_state)
            MPI.Gather!(local_step_states |> Main.data, step_states |> Main.data, MPI.COMM_WORLD)
            MPI.Gather!(local_step_infos |> Main.data, step_info |> Main.data, MPI.COMM_WORLD)
            MPI.Gather!(local_step_reward, step_reward, MPI.COMM_WORLD)
            MPI.Gather!(local_step_status, step_status, MPI.COMM_WORLD)

            #Move the CPU arrays to the correct index of the bigger GPU arrays
            Main.Main.lap(lapTimer, :rollout_state_to_gpu)
            states[:, :, t+1] = step_states
            infos[:, :, t] = step_info
            rewards[:, t] = step_reward
            env_status[:, t] = step_status
        end

        return (;states, actor_output, rewards, status=env_status, infos)
    end
end

function BatchCollectorWorker(envs, params)
    action_size =  mapreduce(s->length(s), +, null_action(envs[1], params))

    steps_per_batch = params.rollout.n_steps_per_epoch
    n_envs = params.rollout.n_envs
    n_local_envs = length(envs)
    template_state = ComponentTensor(state(envs[1], params))
    template_info = ComponentTensor(info(envs[1], params))
    local_step_states = BatchComponentTensor(template_state, n_local_envs, array_fcn=(inds...)->zeros(Float32, inds...))
    local_step_reward = zeros(Float32, n_local_envs)
    local_step_status = zeros(UInt8, n_local_envs)
    local_step_infos  = BatchComponentTensor(template_info, n_local_envs, array_fcn=(inds...)->zeros(Float32, inds...))
    local_step_actions = zeros(Float32, action_size, n_local_envs)

    function collect()
        
        #States for the first timestep
        #@Threads.threads 
        for i=1:n_local_envs
            if params.rollout.reset_on_epoch_start
                reset!(envs[i], params)
            end
            local_step_states[:, i] = state(envs[i], params)
        end
        MPI.Gather!(local_step_states |> data, nothing, MPI.COMM_WORLD)
        for t=1:steps_per_batch
            MPI.Scatter!(nothing, local_step_actions, MPI.COMM_WORLD)
            #@Threads.threads 
            for i=1:n_local_envs
                env = envs[i]
                action = view(local_step_actions, :, i)
                act!(env, action, params)
                local_step_states[:, i] = state(envs[i], params)
                local_step_reward[i] = reward(env, params)
                local_step_status[i] = status(env, params)
                local_step_infos[:, i] = info(envs[i], params)
                if local_step_status[i] != RUNNING
                    reset!(env, params)
                end
            end
            MPI.Barrier(MPI.COMM_WORLD)
            MPI.Gather!(local_step_states |> data, nothing, MPI.COMM_WORLD)
            MPI.Gather!(local_step_infos |> data, nothing, MPI.COMM_WORLD)
            MPI.Gather!(local_step_reward, nothing, MPI.COMM_WORLD)
            MPI.Gather!(local_step_status, nothing, MPI.COMM_WORLD)
        end
        return nothing
    end
end

function compute_batch_stats(batch)
    logdict = Dict{String, Number}()
    merge!(logdict, quantile_dict("actor/mus", view( batch.actor_output, :mu, :, :)))
    merge!(logdict, quantile_dict("actor/sigmas", view( batch.actor_output, :sigma, :, :)))
    merge!(logdict, quantile_dict("actor/action_ctrl", view( batch.actor_output, :action, :, :)))
    merge!(logdict, quantile_dict("actor/action_ctrl_sum_squared", sum(view(batch.actor_output, :action, :, :).^2; dims=1)))
    merge!(logdict, quantile_dict("actor/latent", view(batch.actor_output, :latent, :, :)))
    merge!(logdict, quantile_dict("actor/latent_mu", view(batch.actor_output, :latent_mu, :, :)))
    merge!(logdict, quantile_dict("actor/latent_logsigma", view(batch.actor_output, :latent_logsigma, :, :)))
    merge!(logdict, quantile_dict("rollout_batch/rewards", batch.rewards))
    logdict["rollout_batch/termination_rate"] = sum(batch.status .== Main.MuJoCoEnvs.TERMINATED) / length(batch.status)
    batch_done = Array(batch.status .!= Main.MuJoCoEnvs.RUNNING)
    merge!(logdict, quantile_dict("rollout_batch/lifespan", view(Main.array(batch.infos.lifetime), 1, :, :)[batch_done]))
    for key in keys(Main.index(batch.infos))
        merge!(logdict, quantile_dict("rollout_batch/$key", batch.infos[key]))
    end
    return logdict
end
