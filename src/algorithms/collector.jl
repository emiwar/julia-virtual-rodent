include("../utils/named_batch_tuple.jl")

function collect_batch(envs, actor_critic, params)
    steps_per_batch = params.n_steps_per_batch
    n_envs = length(envs)

    #Big GPU arrays/NamedBatchTuples for storing the entire batch
    states = gpu_zeros_from_template(state(envs[1], params), (n_envs, steps_per_batch+1))
    actor_output_template = actor(actor_critic, view(states, 1, 1), params)
    actor_output = gpu_zeros_from_template(actor_output_template, (n_envs, steps_per_batch))
    rewards = CUDA.zeros(        n_envs, steps_per_batch)
    terminated = CUDA.zeros(Bool,n_envs, steps_per_batch)
    #...except infos, which never have to be moved to the GPU
    infos = cpu_zeros_from_template(info(envs[1]), (n_envs, steps_per_batch))

    #Smaller arrays/NamedBatchTuples for keeping one timestep in CPU memory while multithreading
    step_states = cpu_zeros_from_template(state(envs[1], params), (n_envs,))
    step_reward = zeros(Float32, n_envs)
    step_terminated = zeros(Bool, n_envs)

    #States for the first timestep
    @Threads.threads for i=1:n_envs
        if params.reset_epoch_start
            reset!(envs[i])
        end
        step_states[i] = state(envs[i], params)
    end
    states[:, 1] = step_states

    for t=1:steps_per_batch
        actor_output[:, t] = actor(actor_critic, view(states, :, t), params)
        step_actions = Array(view(actor_output.action, :, :, t))#NamedBatchTuple(map(Flux.cpu, view(actor_output, :, t).action))
        @Threads.threads for i=1:n_envs
            env = envs[i]
            action = view(step_actions, :, i)
            act!(env, action, params)
            step_states[i] = state(env, params)
            step_reward[i] = reward(env, params)
            step_terminated[i] = is_terminated(env, params)
            infos[i, t] = info(env)
            if step_terminated[i]
                reset!(env)
            end
        end
        #Move the CPU arrays to the correct index of the bigger GPU arrays
        states[:, t+1] = step_states
        rewards[:, t] = step_reward
        terminated[:, t] = step_terminated
    end

    return (;states, actor_output, rewards, terminated, infos)
end
