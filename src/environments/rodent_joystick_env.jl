
abstract type RodentFollowEnv <: MuJoCoEnv end

mutable struct RodentJoystickEnv <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    control_t::Float64
    lifetime::Int64
    cumulative_reward::Float64
    sensorranges::Dict{String, UnitRange{Int64}}
end

function RodentJoystickEnv(params; target_data="src/environments/assets/diego_curated_snippets.h5")
    if params.physics.body_scale != 1.0
        @warn "Code for rescaling imitation target removed from preprocessing, will use imitation target for scale=1.0." params.physics.body_scale
    end
    model = dm_control_rodent(torque_actuators = params.physics.torque_control,
                              foot_mods = params.physics.foot_mods,
                              scale = params.physics.body_scale,
                              hip_mods = params.physics.hip_mods,
                              physics_timestep = params.physics.timestep,
                              control_timestep = params.physics.timestep * params.physics.n_physics_steps)
    data = MuJoCo.init_data(model)
    sensorranges = prepare_sensorranges(model, "walker/" .* ["accelerometer", "velocimeter",
                                                             "gyro", "palm_L", "palm_R",
                                                             "sole_L", "sole_R", "torso"])
    env = RodentJoystickEnv(model, data, 0.0, 0, 0.0, sensorranges)
    reset!(env, params)
    return env
end

function clone(env::RodentJoystickEnv, params)
    new_env = RodentJoystickEnv(
        env.model,
        MuJoCo.init_data(env.model),
        0.0, 0, 0.0,env.sensorranges
    )
    reset!(new_env, params)
    return new_env
end

function state(env::RodentJoystickEnv, params)
    (
        proprioception = (
            joints = (@view env.data.qpos[8:end]),
            joint_vel = (@view env.data.qvel[7:end]),
            actuations = (@view env.data.act[:]),
            head = (
                velocity = sensor(env, "walker/velocimeter"),
                accel = sensor(env, "walker/accelerometer"),
                gyro = sensor(env, "walker/gyro")
            ),
            torso = (
                velocity = sensor(env, "walker/torso"),
                xmat = reshape(body_xmat(env, "walker/torso"), :),
                com = subtree_com(env, "walker/torso")
            ),
            paw_contacts = (
                palm_L = sensor(env, "walker/palm_L"),
                palm_R = sensor(env, "walker/palm_R"),
                sole_L = sensor(env, "walker/sole_L"),
                sole_R = sensor(env, "walker/sole_R")
            )
        ),
        command = (
            forward_speed = target_forward_speed(env, params),
            turning_speed = target_turning_speed(env, params)
        )
    )
end

function reward(env::RodentJoystickEnv, params)
    forward_falloff = params.reward.falloff.forward_speed^2
    forward_reward = exp(-(forward_speed(env) - target_forward_speed(env, params))^2/forward_falloff)
    turn_falloff = params.reward.falloff.turning_speed^2
    turn_reward = exp(-(turning_speed(env) - target_turning_speed(env, params))^2/turn_falloff)
    ctrl_reward = -params.reward.control_cost * norm(env.data.ctrl)^2
    total_reward  = forward_reward + turn_reward
    total_reward += ctrl_reward + params.reward.alive_bonus
    total_reward #return clamp(total_reward, params.min_reward, Inf)
end

function status(env::RodentJoystickEnv, params)
    if torso_z(env) < params.physics.min_torso_z 
        return TERMINATED
    else
        return RUNNING
    end
end

function info(env::RodentJoystickEnv, params)
    (
        torso_x=torso_x(env),
        torso_y=torso_y(env),
        torso_z=torso_z(env),
        lifetime=float(env.lifetime),
        cumulative_reward = env.cumulative_reward,
        actuator_force_sum_sqr = norm(env.data.actuator_force)^2,
        forward_speed = forward_speed(env),
        turning_speed = turning_speed(env)
    )
end

#Actions
function act!(env::RodentJoystickEnv, action, params)
    env.data.ctrl .= clamp.(action, -1.0, 1.0)
    for _=1:params.physics.n_physics_steps
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    env.control_t += 1.0
    env.cumulative_reward += reward(env, params)
end

function reset!(env::RodentJoystickEnv, params)
    env.lifetime = 0
    env.control_t = 0.0
    env.cumulative_reward = 0.0
    MuJoCo.reset!(env.model, env.data)
end

#Utils
torso_x(env::RodentFollowEnv) = subtree_com(env, "walker/torso")[1]
torso_y(env::RodentFollowEnv) = subtree_com(env, "walker/torso")[2]
torso_z(env::RodentFollowEnv) = subtree_com(env, "walker/torso")[3]

function forward_speed(env::RodentJoystickEnv)
    allocentric_v = SVector(env.data.qvel[1], env.data.qvel[2], 0.0)
    egocentric_v  = body_xmat(env, "walker/torso") * allocentric_v
    return egocentric_v[1] #Is x forward? Or should it be y?
end

function turning_speed(env::RodentJoystickEnv)
    return env.data.qvel[6] #This should be turning in global z?
end

target_forward_speed(env::RodentJoystickEnv, params) = 0.2
target_turning_speed(env::RodentJoystickEnv, params) = 0.0

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
