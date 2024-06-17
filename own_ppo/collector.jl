include("named_batch_tuple.jl")

function collect_batch(envs, actor_critic, params; logfcn=nothing)
    steps_per_batch = params.n_steps_per_batch
    n_envs = length(envs)
    states = gpu_zeros_from_template(state(envs[1], params), (n_envs, steps_per_batch+1))
    actions = gpu_zeros_from_template(null_action(envs[1], params), (n_envs, steps_per_batch))
    loglikelihoods = CUDA.zeros( n_envs, steps_per_batch)
    rewards = CUDA.zeros(        n_envs, steps_per_batch)
    terminated = CUDA.zeros(Bool,n_envs, steps_per_batch)
    sigmas = CUDA.zeros(action_size(actor_critic),  n_envs, steps_per_batch)
    mus =    CUDA.zeros(action_size(actor_critic),  n_envs, steps_per_batch)
    infos = cpu_zeros_from_template(info(envs[1]), (n_envs, steps_per_batch))

    step_states = cpu_zeros_from_template(state(envs[1], params), (n_envs,))
    step_reward = zeros(Float32, n_envs)
    step_terminated = zeros(Bool, n_envs)
    @Threads.threads for i=1:n_envs
        if params.reset_epoch_start
            reset!(envs[i])
        end
        step_states[i] = state(envs[i], params)
    end
    states[:, 1] = step_states
    for t=1:steps_per_batch
        actor_output = actor(actor_critic, view(states, :, t), params)
        mus[:, :, t] .= actor_output.mu
        sigmas[:, :, t] .= actor_output.sigma
        actions[:, t] = actor_output.action
        loglikelihoods[:, t] .= actor_output.loglikelihood
        step_actions = NamedBatchTuple(map(Flux.cpu, actor_output.action))
        @Threads.threads for i=1:n_envs
            env = envs[i]
            action = view(step_actions, i)
            act!(env, action, params)
            step_states[i] = state(env, params)
            step_reward[i] = reward(env, params)
            step_terminated[i] = is_terminated(env, params)
            infos[i, t] = info(env)
            if step_terminated[i]
                reset!(env)
            end
        end
        states[:, t+1] = step_states
        rewards[:, t] = step_reward
        terminated[:, t] = step_terminated
    end
    if !isnothing(logfcn)
        logfcn("actor/mus", mus)
        logfcn("actor/sigmas", sigmas)
        logfcn("actor/action_ctrl", actions.ctrl)
        logfcn("actor/action_ctrl_sum_squared", sum(actions.ctrl.^2; dims=1))
        logfcn("rollout_batch/rewards", rewards)
        logfcn("rollout_batch/failure_rate", sum(terminated) / length(terminated))
        for (key, val) in infos |> pairs
            logfcn("rollout_batch/$key", val)
        end
    end  

    return (;states, actions, loglikelihoods, rewards, terminated, info)
end
