
mutable struct RodentJoystickEnv <: MuJoCoEnv# <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    control_t::Float64
    last_torso_pos::SVector{3, Float64}
    last_torso_quat::SVector{4, Float64}
    lifetime::Int64
    cumulative_reward::Float64
    sensorranges::Dict{String, UnitRange{Int64}}
end

function RodentJoystickEnv(params)
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
    env = RodentJoystickEnv(model, data, 0.0,
                            SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0, 0.0),
                            0, 0.0, sensorranges)
    reset!(env, params)
    return env
end

function clone(env::RodentJoystickEnv, params)
    new_env = RodentJoystickEnv(
        env.model,
        MuJoCo.init_data(env.model), 0.0,
        env.last_torso_pos, env.last_torso_quat,
        0, 0.0, env.sensorranges
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
        ),
        for_critic_only = (
            forward_speed = forward_speed(env, params),
            turning_speed = turning_speed(env, params)
        )
    )
end


function forward_reward(env::RodentJoystickEnv, params)
    falloff = params.reward.falloff.forward_speed^2
    reward = exp(-(forward_speed(env, params) - target_forward_speed(env, params))^2/falloff)
    #speed  = forward_speed(env, params)
    #target = params.reward.target.forward_speed
    #1 - (speed - target)^2/target^2
    return reward
end
function turning_reward(env::RodentJoystickEnv, params)
    falloff = params.reward.falloff.turning_speed^2
    reward = exp(-(turning_speed(env, params) - target_turning_speed(env, params))^2/falloff)
    #target = params.reward.target.turning_speed
    return reward #1 - (speed - target)^2
end

function reward(env::RodentJoystickEnv, params)
    total_reward  = forward_reward(env, params) + turning_reward(env, params)
    ctrl_reward   = -params.reward.control_cost * norm(env.data.ctrl)^2
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
        forward_speed = forward_speed(env, params),
        turning_speed = turning_speed(env, params),
        forward_reward = forward_reward(env, params),
        turning_reward = turning_reward(env, params)
    )
end

#Actions
function act!(env::RodentJoystickEnv, action, params)
    env.last_torso_pos  = body_xpos(env,  "walker/torso")
    env.last_torso_quat = body_xquat(env, "walker/torso")
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
    MuJoCo.forward!(env.model, env.data)
    env.last_torso_pos  = body_xpos(env,  "walker/torso")
    env.last_torso_quat = body_xquat(env, "walker/torso")    
end

#Utils
torso_x(env::RodentJoystickEnv) = subtree_com(env, "walker/torso")[1]
torso_y(env::RodentJoystickEnv) = subtree_com(env, "walker/torso")[2]
torso_z(env::RodentJoystickEnv) = subtree_com(env, "walker/torso")[3]

function forward_speed(env::RodentJoystickEnv, params)
    timestep::Float64 = env.model.opt.timestep * params.physics.n_physics_steps
    vec = (body_xpos(env, "walker/torso") - env.last_torso_pos) ./ timestep
    allocentric_v = SVector(vec[1], vec[2], 0.0)
    egocentric_v  = body_xmat(env, "walker/torso") * allocentric_v
    return egocentric_v[1] #Is x forward? Or should it be y?
end

function turning_speed(env::RodentJoystickEnv, params)
    timestep::Float64 = env.model.opt.timestep * params.physics.n_physics_steps
    az_diff = azimuth_between(env.last_torso_quat, body_xquat(env, "walker/torso"))
    return az_diff / timestep
end

target_forward_speed(env::RodentJoystickEnv, params) = params.reward.target.forward_speed
target_turning_speed(env::RodentJoystickEnv, params) = params.reward.target.turning_speed

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
