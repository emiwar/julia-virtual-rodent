include("mujoco_env.jl")
mutable struct RodentRunEnv <: MuJoCoEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    last_torso_x::Float64
    lifetime::Int64
    cumulative_reward::Float64
    max_life::Int64
    sensorranges::Dict{String, UnitRange{Int64}}
end

function RodentRunEnv(params)
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
    env = RodentRunEnv(model, data, 0.0, 0, 0.0, 1000, sensorranges)
    reset!(env, params)
    return env
end

function clone(env::RodentRunEnv, params)
    new_env = RodentRunEnv(
        env.model,
        MuJoCo.init_data(env.model),
        0.0, 0, 0.0, 1000, env.sensorranges
    )
    reset!(new_env, params)
    return new_env
end

#Read-outs
function state(env::RodentRunEnv, params)
    (;
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
        )
    )
end

function reward(env::RodentRunEnv, params)
    forward_reward = params.reward.forward_weight * torso_speed_x(env)
    ctrl_reward = -params.reward.control_cost * norm(env.data.ctrl)^2
    total_reward = forward_reward + ctrl_reward + params.reward.alive_bonus
    return total_reward 
end

function status(env::RodentRunEnv, params)
    if torso_z(env) < params.physics.min_torso_z 
        return TERMINATED
    elseif env.lifetime > env.max_life
        return TRUNCATED
    else
        return RUNNING
    end
end

function info(env::RodentRunEnv, params)
    (
        torso_x=torso_x(env),
        torso_y=torso_y(env),
        torso_z=torso_z(env),
        lifetime=float(env.lifetime),
        cumulative_reward = env.cumulative_reward,
        actuator_force_sum_sqr = norm(env.data.actuator_force)^2,
        forward_speed = torso_speed_x(env)
    )
end

#Actions
function act!(env::RodentRunEnv, action, params)
    env.last_torso_x = torso_x(env)
    env.data.ctrl .= clamp.(action, -1.0, 1.0)
    for _=1:params.physics.n_physics_steps
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    env.cumulative_reward += reward(env, params)
end

function reset!(env::RodentRunEnv, params)
    MuJoCo.reset!(env.model, env.data)
    MuJoCo.forward!(env.model, env.data) #Run model forward to get correct initial state
    env.last_torso_x = torso_x(env)
    env.lifetime = 0
    env.cumulative_reward = 0.0
    env.max_life = rand(1000:2000)
end

#Utils
torso_x(env::MuJoCoEnv) = subtree_com(env, "walker/torso")[1]
torso_y(env::MuJoCoEnv) = subtree_com(env, "walker/torso")[2]
torso_z(env::MuJoCoEnv) = subtree_com(env, "walker/torso")[3]
torso_speed_x(env::RodentRunEnv) = (torso_x(env) - env.last_torso_x) / env.model.opt.timestep
