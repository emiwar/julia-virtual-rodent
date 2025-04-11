mutable struct RodentDecoderRunEnv{D} <: MuJoCoEnv
    rodent::Rodent
    decoder::D
    last_x_com::Float64
    target_speed::Float64
    lifetime::Int64
    cumulative_reward::Float64
end

function RodentDecoderRunEnv(params)
    rodent = Rodent(params)
    params, weights_file_name = load_from_wandb(params.run.decoder_wandb_run_id, r"step-.*")
    actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu
    env = RodentDecoderRunEnv(rodent, actor_critic, 0.0, 0.0, 0, 0.0)
    reset!(env, params)
    return env
end

function clone(env::RodentDecoderRunEnv, params)
    new_env = RodentDecoderRunEnv(clone(env.rodent), env.decoder, 0.0, 0.0, 0, 0.0)
    reset!(new_env, params)
    return new_env
end

state(env::RodentDecoderRunEnv, params) = proprioception(env.rodent)

function forward_speed(env::RodentDecoderRunEnv)
    dt = params.physics.timestep * params.physics.n_physics_steps
    v  = (env.last_x_com - torso_x(env.rodent)) / dt
end

function forward_reward(env::RodentDecoderRunEnv, params)
    exp(-(forward_speed(env) - env.target_speed)^2 / (params.reward.falloff.speed^2))
end

function reward(env::RodentDecoderRunEnv, params)
    r = forward_reward(env, params)
    r -= params.reward.control_cost * norm(env.data.ctrl)^2
    r += params.reward.alive_bonus
    if haskey(params.reward, :energy_cost)
        r -= params.reward.energy_cost * energy_cost(env)
    end
    r
end

function status(env::RodentDecoderRunEnv, params)
    if torso_z(env.rodent) < params.physics.min_torso_z 
        return TERMINATED
    elseif env.lifetime > params.run.episode_length
        return TRUNCATED
    else
        return RUNNING
    end
end

function info(env::RodentDecoderRunEnv, params)
    (
        qpos_root=(@view env.rodent.data.qpos[1:7]),
        torso_x=torso_x(env.rodent),
        torso_y=torso_y(env.rodent),
        torso_z=torso_z(env.rodent),
        lifetime=float(env.lifetime),
        cumulative_reward = env.cumulative_reward,
        actuator_force_sum_sqr = norm(env.rodent.data.actuator_force)^2,
        forward_speed = forward_speed(env),
        forward_reward = forward_reward(env, params),
    )
end

#Actions
function act!(env::RodentDecoderRunEnv, action, params)
    env.last_x_com = torso_x(env.rodent)
    env.data.ctrl .= clamp.(action, -1.0, 1.0)
    for _=1:params.physics.n_physics_steps
        step!(env.rodent)
    end
    env.lifetime += 1
    env.cumulative_reward += reward(env, params)
end

function reset!(env::RodentDecoderRunEnv, params, next_clip, next_frame)
    env.lifetime = 0
    env.cumulative_reward = 0.0
    env.target_speed = params.run.min_target_speed + rand() * (params.run.max_target_speed - params.run.min_target_speed)
    reset!(env.rodent)
end

function reset!(env::RodentDecoderRunEnv, params)
    if params.imitation.restart_on_reset
        reset!(env, params, rand(1:(size(env.target)[3])), 1)
    else
        reset!(env, params, env.target_clip, env.target_frame)
    end
end

function preprocess_actions(envs::Vector{E}, actions, state, reset_mask, params) where E <: RodentDecoderRunEnv
    actor_critic = first(envs).decoder

    proprioception = Flux.ignore(()->state |> array |> copy)

    decoder_input = cat(actions, proprioception; dims=1)
    decoder_output = rollout!(actor_critic.decoder, decoder_input, reset_mask)

    action_and_loglikelihood = actor_critic.action_sampler(decoder_output, action)

    return action_and_loglikelihood.mu
end

function energy_cost(env::RodentDecoderRunEnv)
    mapreduce((v,f) -> abs(v)*abs(f), +,
              (@view env.rodent.data.qvel[7:end]),
              (@view env.rodent.data.qfrc_actuator[7:end]))
end

function Base.show(io::IO, env::RodentDecoderRunEnv)
    compact = get(io, :compact, false)
    if compact
        print(io, "RodentDecoderRunEnv")
    else
        println(io, "RodentDecoderRunEnv:")
        println(io, "\tlifetime: $(env.lifetime)")
        println(io, "\tcumulative_reward: $(env.cumulative_reward)")
    end
end
