include("../utils/component_tensor.jl")

function collect_batch(envs, actor_critic, params)
    steps_per_batch = params.n_steps_per_batch
    n_envs = length(envs)

    #Big GPU arrays/BatchComponentTensor for storing the entire batch
    template_state = state(envs[1], params)
    states = BatchComponentTensor(template_state, n_envs, steps_per_batch+1; array_fcn=CUDA.zeros)
    template_actor_output = actor(actor_critic, view(states, :, 1, 1), params) |> ComponentTensor
    actor_output =  BatchComponentTensor(template_actor_output, n_envs, steps_per_batch; array_fcn=CUDA.zeros)
    rewards = CUDA.zeros(        n_envs, steps_per_batch)
    env_status = CUDA.zeros(UInt8, n_envs, steps_per_batch)

    #...except infos, which never have to be moved to the GPU
    template_info = info(envs[1])
    infos = BatchComponentTensor(template_info, n_envs, steps_per_batch; array_fcn=zeros)

    #Smaller arrays/ComponentTensor for keeping one timestep in CPU memory while multithreading
    step_states = BatchComponentTensor(template_state, n_envs)
    step_reward = zeros(Float32, n_envs)
    step_status = zeros(UInt8, n_envs)

    #States for the first timestep
    @Threads.threads for i=1:n_envs
        if params.reset_epoch_start
            reset!(envs[i], params)
        end
        step_states[:, i] = state(envs[i], params)
    end
    states[:, :, 1] = step_states
    for t=1:steps_per_batch
        actor_output[:, :, t] = actor(actor_critic, view(states, :, :, t), params) |> ComponentTensor
        step_actions = Array(view(actor_output, :action, :, t))
        @Threads.threads for i=1:n_envs
            env = envs[i]
            action = view(step_actions, :, i)
            act!(env, action, params)
            step_states[:, i] = state(env, params)
            step_reward[i] = reward(env, params)
            step_status[i] = status(env, params)
            infos[:, i, t] = info(env)
            if step_status[i] != RUNNING
                reset!(env, params)
            end
        end
        #Move the CPU arrays to the correct index of the bigger GPU arrays
        states[:, :, t+1] = step_states
        rewards[:, t] = step_reward
        env_status[:, t] = step_status
    end

    return (;states, actor_output, rewards, status=env_status, infos)
end
