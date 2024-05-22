mutable struct MultiThreadedMuJoCo{T <: MuJoCoEnv}
    model::MuJoCo.Model
    envs::Vector{T}
    n_steps_taken::Vector{Int64}
    episode_reward::Vector{Float64}
    actions::Matrix{Float32}
    states::Matrix{Float32}
    rewards::Vector{Float32}
    terminated::Vector{Bool}

    n_terminated_episodes::Threads.Atomic{Int64}
    sum_episode_reward::Threads.Atomic{Float64}
    sum_episode_length::Threads.Atomic{Int64}

    n_steps::Int64
    sum_epoch_reward::Float64
    sum_epoch_x_speed::Float64
    sum_epoch_x_pos::Float64
end

function MultiThreadedMuJoCo{T}(model::MuJoCo.Model, n_envs::Integer) where T <: MuJoCoEnv
    action_size = MuJoCo.mj_stateSize(model, MuJoCo.LibMuJoCo.mjSTATE_CTRL)
    state_size = MuJoCo.mj_stateSize(model, MuJoCo.LibMuJoCo.mjSTATE_PHYSICS)
    MultiThreadedMuJoCo(
        model,
        [T(model) for _=1:n_envs],
        zeros(Int64, n_envs),
        zeros(Float64, n_envs),
        zeros(Float32, action_size, n_envs),
        zeros(Float32, state_size, n_envs),
        zeros(Float32, n_envs),
        zeros(Bool, n_envs),
        Threads.Atomic{Int64}(0),
        Threads.Atomic{Float64}(0.0),
        Threads.Atomic{Int64}(0),
        0, 0.0, 0.0, 0.0
    )
end

n_envs(env::MultiThreadedMuJoCo) = length(env.envs)
state_size(env::MultiThreadedMuJoCo) = size(env.states, 1)
action_size(env::MultiThreadedMuJoCo) = size(env.actions, 1)

function prepare_batch!(env::MultiThreadedMuJoCo, params)
    env.n_terminated_episodes[] = 0
    env.sum_episode_reward[] = 0.0
    env.sum_episode_length[] = 0
    env.n_steps = 0
    env.sum_epoch_reward = 0.0
    env.sum_epoch_x_speed = 0.0
    env.sum_epoch_x_pos = 0.0
    if params.reset_epoch_start
        for i=1:n_envs(env)
            env.n_steps_taken[i] = 0
            env.episode_reward[i] = 0.0
            reset!(env.model, env.envs[i])
        end
    end
    for i=1:n_envs(env)
        env.states[:, i] .= state(env.model, env.envs[i])
    end
end

function step!(env::MultiThreadedMuJoCo, params)
    step_x_speed = zeros(n_envs(env))
    step_x_pos = zeros(n_envs(env))
    @Threads.threads for i=1:n_envs(env)
        act!(env.model, env.envs[i], view(env.actions, :, i), params)
        env.states[:, i] .= state(env.model, env.envs[i])
        env.rewards[i] = reward(env.model, env.envs[i], params)
        env.terminated[i] = is_terminated(env.envs[i], params)
        env.n_steps_taken[i] += 1
        env.episode_reward[i] += env.rewards[i]
        step_x_pos[i] = torso_x(env.envs[i])
        step_x_speed[i] = torso_speed_x(env.envs[i])
        if env.terminated[i]
            Threads.atomic_add!(env.n_terminated_episodes, 1)
            Threads.atomic_add!(env.sum_episode_reward, env.episode_reward[i])
            Threads.atomic_add!(env.sum_episode_length, env.n_steps_taken[i])
            env.n_steps_taken[i] = 0
            env.episode_reward[i] = 0.0
            reset!(env.model, env.envs[i])
        end
    end
    env.n_steps += n_envs(env)
    env.sum_epoch_reward += sum(env.rewards)
    env.sum_epoch_x_speed += sum(step_x_speed)
    env.sum_epoch_x_pos += sum(step_x_pos)
end

function stats(env::MultiThreadedMuJoCo)
    (;n_terminated_episodes = env.n_terminated_episodes[],
      episode_avg_reward = env.sum_episode_reward[] / env.n_terminated_episodes[],
      episode_avg_length = env.sum_episode_length[] / env.n_terminated_episodes[],
      epoch_avg_reward = env.sum_epoch_reward / n_envs(env),
      epoch_avg_x_pos = env.sum_epoch_x_pos / env.n_steps,
      epoch_avg_x_speed = env.sum_epoch_x_speed / env.n_steps
    )
    #epoch_avg_z
end