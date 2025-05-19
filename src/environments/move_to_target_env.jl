mutable struct MoveToTargetEnv{W} <: AbstractEnv# <: RodentFollowEnv
    walker::W
    control_cost::Float64
    alive_bonus::Float64
    target::MVector{3, Float64}
    lifetime::Int64
    cumulative_reward::Float64
end

function MoveToTargetEnv(walker, control_cost, alive_bonus)
    env = MoveToTargetEnv(walker, control_cost, alive_bonus,
                          MVector(0.0, 0.0, 0.0), 0, 0.0)
    reset!(env)
    return env
end

function reward(env::MoveToTargetEnv)
    control_reward = -env.control_cost * norm(env.walker.data.ctrl)^2
    closeness_reward = clamp(1.0 - dist_to_target(env), 0.0, 1.0)
    return closeness_reward + control_reward + env.alive_bonus
end

function state(env::MoveToTargetEnv)
    (proprioception = proprioception(env.walker),
     target_vector = target_vector(env))
end

duplicate(env::MoveToTargetEnv) = MoveToTargetEnv(clone(env.walker), env.control_cost, env.alive_bonus)

function status(env::MoveToTargetEnv)
    if torso_z(env.walker) < min_torso_z(env.walker)
        return TERMINATED
    elseif env.lifetime >= 1000 || dist_to_target(env) <= 0.01
        return TRUNCATED
    else
        return RUNNING
    end
end

function info(env::MoveToTargetEnv)
    (
        info(env.walker)...,   
        lifetime = float(env.lifetime),
        cumulative_reward = env.cumulative_reward,
        dist_to_target = dist_to_target(env),
        dist_from_start = norm(subtree_com(env.walker, "walker/torso")),
        torso_height = subtree_com(env.walker, "walker/torso")[3],
        head_height = subtree_com(env.walker, "walker/skull")[3]
    )
end

#Actions
function act!(env::MoveToTargetEnv, action)
    set_ctrl!(env.walker, action)
    for _=1:env.walker.n_physics_steps
        step!(env.walker)
    end
    env.cumulative_reward += reward(env)
    env.lifetime += 1
end

function reset!(env::MoveToTargetEnv)
    env.lifetime = 0
    env.cumulative_reward = 0.0
    env.target .= random_target(env)
    reset!(env.walker)
end

null_action(env::MoveToTargetEnv) = null_action(env.walker)

function dist_to_target(env::MoveToTargetEnv)
    norm(subtree_com(env.walker, "walker/torso") - env.target)
end

function random_target(env::MoveToTargetEnv)
    SVector(0.5*randn(), 0.5*randn(), 0.04 + 0.06*rand())
end

function target_vector(env::MoveToTargetEnv)
    body_xmat(env.walker, "walker/torso") * (-(subtree_com(env.walker, "walker/torso") - env.target))
end

function Base.show(io::IO, env::MoveToTargetEnv)
    compact = get(io, :compact, false)
    if compact
        print(io, "MoveToTargetEnv{Walker=$(env.walker)}")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)MoveToTargetEnv")
        indented_io = IOContext(io, :indent => (get(io, :indent, 0) + 2))
        show(indented_io, env.walker)
        println(io, "$(indent)lifetime: $(env.lifetime)")
        println(io, "$(indent)cumulative_reward: $(env.cumulative_reward)")
    end
end
