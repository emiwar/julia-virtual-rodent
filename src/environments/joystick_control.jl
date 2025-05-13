mutable struct RodentJoystickEnv{W, F <:NamedTuple, C <: NamedTuple} <: AbstractEnv# <: RodentFollowEnv
    walker::W
    falloffs::F
    command::C
    last_torso_pos::SVector{3, Float64}
    last_torso_quat::SVector{4, Float64}
    time_since_command::Float64
    lifetime::Int64
    cumulative_reward::Float64
end

function RodentJoystickEnv(walker, falloffs)
    env = RodentJoystickEnv(walker, falloffs, random_joystick_command(),
                            (@SVector zeros(3)), (@SVector zeros(4)), 0.0, 0, 0.0)
    reset!(env)
    return env
end

function reward(env::RodentJoystickEnv)
    sum(compute_rewards(env)) #alive bonus and control cost?
end

function state(env::RodentJoystickEnv)
    (proprioception = proprioception(walker),
     command = env.command)
end

duplicate(env::RodentJoystickEnv) = RodentJoystickEnv(clone(env.walker), env.falloffs)

function status(env::RodentJoystickEnv)
    if torso_z(env.walker) < min_torso_z(env.walker)
        return TERMINATED
    else
        return RUNNING
    end
end

function info(env::RodentJoystickEnv)
    (
        info(env.walker)...,   
        lifetime = float(env.lifetime),
        cumulative_reward = env.cumulative_reward,
        forward_speed = forward_speed(env),
        turning_speed = turning_speed(env),
        head_height   = head_height(env),
        command = env.command,
        rewards = compute_rewards(env),
    )
end

#Actions
function act!(env::RodentJoystickEnv, action)
    env.last_torso_pos  = body_xpos(env.walker,  "walker/torso")
    env.last_torso_quat = body_xquat(env.walker, "walker/torso")
    set_ctrl!(env.walker, action)
    for _=1:env.walker.n_physics_steps
        step!(env.walker)
    end
    env.cumulative_reward += reward(env)
    env.lifetime += 1
    env.time_since_command += 1.0
    if env.time_since_command >= 200.0
        #This is actually a bit unfair since reward for the rest of this step will be calculated
        #based on the new command, wheras the network acted on the old one. Hmm...
        env.command = random_joystick_command()
        env.time_since_command = 0.0
    end
end

function reset!(env::RodentJoystickEnv)
    env.lifetime = 0
    env.cumulative_reward = 0.0
    env.time_since_command = 0.0
    env.command = random_joystick_command()
    reset!(env.walker)
    env.last_torso_pos  = body_xpos(env.walker,  "walker/torso")
    env.last_torso_quat = body_xquat(env.walker, "walker/torso")    
end

function compute_rewards(env::RodentJoystickEnv)
    current = (forward = forward_speed(env),
               turning = turning_speed(env),
               head    = head_height(env))
    targets = get_command(env)
    rewards = map((c,t,f)->exp(-((c-t)/f)^2), current, targets, env.falloffs)
    return rewards
end

# Speed calulations
#(finite difference between env tranistions might be more robust than MuJoCo props)
function forward_speed(env::RodentJoystickEnv)
    timestep = dt(env.walker) * env.walker.n_physics_steps
    vec = (body_xpos(env.walker, "walker/torso") - env.last_torso_pos) ./ timestep
    allocentric_v = SVector(vec[1], vec[2], 0.0)
    egocentric_v  = body_xmat(env.walker, "walker/torso") * allocentric_v
    return egocentric_v[1] #Is x forward? Or should it be y?
end

function turning_speed(env::RodentJoystickEnv)
    timestep = dt(env.walker) * env.walker.n_physics_steps
    az_diff = azimuth_between(env.last_torso_quat, body_xquat(env, "walker/torso"))
    return az_diff / timestep
end

head_height(env::RodentJoystickEnv) = subtree_com(env.walker, "walker/skull")[3]

# Commands
function random_joystick_command()
    speed = -0.1 + 0.7*rand()
    turning_speed = speed > 0.25 ? 0.0 : -0.5 + rand()
    if abs(speed) < 0.05 && abs(turning_speed) < 0.1
        head_height = 0.08 + 0.05*rand()
    else
        head_height = 0.05 + 0.03*rand()
    end
    return (foward=speed, turning=turning_speed, head=head_height)
end

function Base.show(io::IO, env::RodentJoystickEnv)
    compact = get(io, :compact, false)
    if compact
        print(io, "RodentJoystickEnv")
    else
        println(io, "RodentJoystickEnv:")
        println(io, "\tlifetime: $(env.lifetime)")
        println(io, "\tcumulative_reward: $(env.cumulative_reward)")
    end
end
