include("mujoco_env.jl")
mutable struct RodentEnv <: MuJoCoEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    last_torso_x::Float64
    lifetime::Int64
    cumulative_reward::Float64
end

function RodentEnv()
    modelPath = "src/environments/assets/rodent_with_floor.xml"
    model = MuJoCo.load_model(modelPath)
    RodentEnv(model)
end

function RodentEnv(model::MuJoCo.Model)
    data = MuJoCo.init_data(model)
    MuJoCo.reset!(model, data)
    env = RodentEnv(model, data, 0.0, 0, 0.0)
    reset!(env)
    return env
end

function clone(env::RodentEnv)
    new_env = RodentEnv(
        env.model,
        MuJoCo.init_data(env.model),
        0.0, 0, 0.0
    )
    reset!(new_env)
    return new_env
end

#Read-outs
function state(env::RodentEnv, params)
    ComponentTensor(
        qpos=reshape(env.data.qpos, :),
        qvel=reshape(env.data.qvel, :),
        act=reshape(env.data.act, :),
        head_accel = read_sensor_value(env, "accelerometer") .* 0.1,
        head_vel = read_sensor_value(env, "velocimeter"),
        head_gyro = read_sensor_value(env, "gyro"),
        paw_contacts = [read_sensor_value(env, "palm_L"); 
                        read_sensor_value(env, "palm_R");
                        read_sensor_value(env, "sole_L");
                        read_sensor_value(env, "sole_R")],
        torso_linvel = read_sensor_value(env, "torso"),
        torso_xmat = MuJoCo.body(env.data, "torso").xmat,
        torso_height = MuJoCo.body(env.data, "torso").com[3] .* 10.0
    )
end

function reward(env::RodentEnv, params)
    forward_reward = params.forward_reward_weight * torso_speed_x(env)
    healthy_reward = params.healthy_reward_weight * (!is_terminated(env, params))
    ctrl_reward = -params.ctrl_reward_weight * sum(env.data.actuator_force.^2)
    return forward_reward + healthy_reward + ctrl_reward
end

function is_terminated(env::RodentEnv, params)
    torso_z(env) < params.min_torso_z
end

function info(env::RodentEnv)
    ComponentTensor(
     torso_x=torso_x(env),
     torso_y=torso_y(env),
     torso_z=torso_z(env),
     torso_speed_x=torso_speed_x(env),
     lifetime=env.lifetime,
     cumulative_reward=env.cumulative_reward,
     actuator_force_sum_sqr=sum(env.data.actuator_force.^2)
    )
end

#Actions
function act!(env::RodentEnv, action, params)
    env.last_torso_x = torso_x(env)
    env.data.ctrl .= clamp.(action, -1.0, 1.0)
    for _=1:params.n_physics_steps
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    env.cumulative_reward += reward(env, params)
end

function reset!(env::RodentEnv)
    MuJoCo.reset!(env.model, env.data)
    MuJoCo.forward!(env.model, env.data) #Run model forward to get correct initial state
    env.last_torso_x = torso_x(env)
    env.lifetime = 0
    env.cumulative_reward = 0.0
end

#Utils
torso_x(env::RodentEnv) = MuJoCo.body(env.data, "torso").com[1]
torso_y(env::RodentEnv) = MuJoCo.body(env.data, "torso").com[2]
torso_z(env::RodentEnv) = MuJoCo.body(env.data, "torso").com[3]
torso_speed_x(env::RodentEnv) = (torso_x(env) - env.last_torso_x) / env.model.opt.timestep
