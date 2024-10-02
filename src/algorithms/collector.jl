include("../utils/component_tensor.jl")

function BatchCollectorRoot(envs, actor_critic, params)#, lapTimer::LapTimer)
    action_size =  mapreduce(s->length(s), +, null_action(envs[1], params))
    steps_per_batch = params.rollout.n_steps_per_epoch
    n_envs = params.rollout.n_envs
    n_local_envs = length(envs)

    #Big GPU arrays/BatchComponentTensor for storing the entire batch
    #lap(lapTimer, :collector_gpu_array_alloc)
    template_state = state(envs[1], params)
    states = BatchComponentTensor(template_state, n_envs, steps_per_batch+1; array_fcn=CUDA.zeros)
    template_actor_output = actor(actor_critic, view(states, :, 1, 1), params) |> ComponentTensor
    actor_output =  BatchComponentTensor(template_actor_output, n_envs, steps_per_batch; array_fcn=CUDA.zeros)
    rewards = CUDA.zeros(        n_envs, steps_per_batch)
    env_status = CUDA.zeros(UInt8, n_envs, steps_per_batch)

    #...except infos, which never have to be moved to the GPU
    template_info = info(envs[1])
    infos = BatchComponentTensor(template_info, n_envs, steps_per_batch; array_fcn=zeros)

    #lap(lapTimer, :collector_cpu_array_alloc)
    #Smaller arrays/ComponentTensor for keeping one timestep in CPU memory while multithreading
    step_states = BatchComponentTensor(template_state, n_envs)
    step_reward = zeros(Float32, n_envs)
    step_status = zeros(UInt8, n_envs)
    step_info  = BatchComponentTensor(template_info, n_envs)
    step_actions = zeros(Float32, action_size, n_envs)

    local_step_states = BatchComponentTensor(template_state, n_local_envs)
    local_step_reward = zeros(Float32, n_local_envs)
    local_step_status = zeros(UInt8, n_local_envs)
    local_step_infos  = BatchComponentTensor(template_info, n_local_envs)
    local_step_actions = zeros(Float32, action_size, n_local_envs)
    function collect(lapTimer::LapTimer)
        lap(lapTimer, :resetting_envs)
        #States for the first timestep
        @Threads.threads for i=1:n_local_envs
            if params.rollout.reset_on_epoch_start
                reset!(envs[i], params)
            end
            local_step_states[:, i] = state(envs[i], params)
        end
        lap(lapTimer, :mpi_gather_first_state)
        MPI.Gather!(local_step_states |> data, step_states |> data, MPI.COMM_WORLD)
        lap(lapTimer, :first_state_to_gpu)
        states[:, :, 1] = step_states
        for t=1:steps_per_batch
            lap(lapTimer, :rollout_actor)
            actor_output[:, :, t] = actor(actor_critic, view(states, :, :, t), params) |> ComponentTensor
            lap(lapTimer, :rollout_action_to_cpu)
            CUDA.copyto!(step_actions, view(actor_output, :action, :, t))
            lap(lapTimer, :mpi_scatter_actions)
            MPI.Scatter!(step_actions, local_step_actions, MPI.COMM_WORLD)
            lap(lapTimer, :rollout_envs)
            @Threads.threads for i=1:n_local_envs
                env = envs[i]
                action = view(local_step_actions, :, i)
                act!(env, action, params)
                local_step_states[:, i] = state(env, params)
                local_step_reward[i] = reward(env, params)
                local_step_status[i] = status(env, params)
                local_step_infos[:, i] = info(env)
                if local_step_status[i] != RUNNING
                    reset!(env, params)
                end
            end
            lap(lapTimer, :mpi_wait_for_workers)
            MPI.Barrier(MPI.COMM_WORLD)
            lap(lapTimer, :mpi_gather_state)
            MPI.Gather!(local_step_states |> data, step_states |> data, MPI.COMM_WORLD)
            MPI.Gather!(local_step_infos |> data, step_info |> data, MPI.COMM_WORLD)
            MPI.Gather!(local_step_reward, step_reward, MPI.COMM_WORLD)
            MPI.Gather!(local_step_status, step_status, MPI.COMM_WORLD)

            #Move the CPU arrays to the correct index of the bigger GPU arrays
            lap(lapTimer, :rollout_state_to_gpu)
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
    template_state = state(envs[1], params)
    template_info = info(envs[1])
    local_step_states = BatchComponentTensor(template_state, n_local_envs)
    local_step_reward = zeros(Float32, n_local_envs)
    local_step_status = zeros(UInt8, n_local_envs)
    local_step_infos  = BatchComponentTensor(template_info, n_local_envs)
    local_step_actions = zeros(Float32, action_size, n_local_envs)

    function collect()
        #States for the first timestep
        @Threads.threads for i=1:n_local_envs
            if params.rollout.reset_on_epoch_start
                reset!(envs[i], params)
            end
            local_step_states[:, i] = state(envs[i], params)
        end
        MPI.Gather!(local_step_states |> data, nothing, MPI.COMM_WORLD)

        for t=1:steps_per_batch
            MPI.Scatter!(nothing, local_step_actions, MPI.COMM_WORLD)
            @Threads.threads for i=1:n_local_envs
                env = envs[i]
                action = view(local_step_actions, :, i)
                act!(env, action, params)
                local_step_states[:, i] = state(env, params)
                local_step_reward[i] = reward(env, params)
                local_step_status[i] = status(env, params)
                local_step_infos[:, i] = info(env)
                if local_step_status[i] != RUNNING
                    reset!(env, params)
                end
            end
            MPI.Barrier(MPI.COMM_WORLD)
            MPI.Gather!(local_step_states |> data, nothing, MPI.COMM_WORLD)
            MPI.Gather!(local_step_infos |> data, nothing, MPI.COMM_WORLD)
            MPI.Gather!(local_step_reward, nothing, MPI.COMM_WORLD)
            MPI.Gather!(local_step_status, nothing, MPI.COMM_WORLD)
        #MPI.Barrier(MPI.COMM_WORLD)
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
    merge!(logdict, quantile_dict("rollout_batch/rewards", batch.rewards))
    logdict["rollout_batch/termination_rate"] = sum(batch.status .== TERMINATED) / length(batch.status)
    batch_done = Array(batch.status .!= RUNNING)
    merge!(logdict, quantile_dict("rollout_batch/lifespan", view(batch.infos.lifetime, 1, :, :)[batch_done]))
    for key in keys(index(batch.infos))
        merge!(logdict, quantile_dict("rollout_batch/$key", batch.infos[key]))
    end
    return logdict
end