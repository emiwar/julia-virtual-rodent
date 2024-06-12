include("spec_utils.jl")

function collect_batch(envs, actor_critic, params; logfcn=nothing)
    steps_per_batch = params.n_steps_per_batch
    n_envs = length(envs)
    states = prealloc_spec(state_space(envs[1]), (n_envs, steps_per_batch+1),
                           Float32, array_fcn=CUDA.zeros)
    actions = prealloc_spec(action_space(envs[1]), (n_envs, steps_per_batch),
                            Float32, array_fcn=CUDA.zeros)
    loglikelihoods = CUDA.zeros( n_envs, steps_per_batch)
    rewards = CUDA.zeros(        n_envs, steps_per_batch)
    terminated = CUDA.zeros(Bool,n_envs, steps_per_batch)                        
    sigmas = CUDA.zeros(action_size(actor_critic),  n_envs, steps_per_batch)
    mus =    CUDA.zeros(action_size(actor_critic),  n_envs, steps_per_batch)
    infos = prealloc_spec(info_space(envs[1]), (n_envs, steps_per_batch))

    step_states = prealloc_spec(state_space(envs[1]), (n_envs,))
    step_reward = zeros(Float32, n_envs)
    step_terminated = zeros(Bool, n_envs)
    @Threads.threads for i=1:n_envs
        if params.reset_epoch_start
            reset!(collector.envs[i])
        end
        for (key, val) in state(envs[i], params) |> pairs
            step_states[key][:, i] .= val
        end
    end
    for (key, val) in step_states |> pairs
        states[key][:, :, 1] = val
    end
    for t=1:steps_per_batch
        actor_output = actor(actor_critic, map(s->view(s, :, :, t), states), params)
        mus[:, :, t] .= actor_output.mu
        sigmas[:, :, t] .= actor_output.sigma
        for (key, val) in actor_output.action |> pairs
            actions[key][:, :, t] .= val
        end
        loglikelihoods[:, t] .= actor_output.loglikelihood
        step_actions = map(Flux.cpu, actor_output.action)
        @Threads.threads for i=1:n_envs
            env = envs[i]
            action = map(a->view(a, :, i), step_actions)
            act!(env, action, params)
            for (key, val) in state(env, params) |> pairs
                step_states[key][:, i] .= val
            end
            step_reward[i] = reward(env, params)
            step_terminated[i] = is_terminated(env, params)
            for (key, val) in info(env) |> pairs
                if val isa AbstractArray
                    infos[key][:, i, t] .= val
                else
                    infos[key][i, t] = val
                end
            end
            if step_terminated[i]
                reset!(env)
            end
        end
        for (key, val) in step_states |> pairs
            states[key][:, :, t+1] = val
        end
        rewards[:, t] = step_reward
        terminated[:, t] = step_terminated
    end
    if !isnothing(logfcn)
        logfcn("actor/mus", mus)
        logfcn("actor/sigmas", sigmas)
        logfcn("actor/action_torques", actions.torques)
        logfcn("actor/action_torques_squared", actions.torques.^2)
        logfcn("rollout_batch/rewards", rewards)
        for (key, val) in infos |> pairs
            logfcn("rollout_batch/$key", val)
        end
    end  

    return (;states, actions, loglikelihoods, rewards, terminated, info)
end
