import MuJoCo

abstract type MuJoCoEnv end

mutable struct RodentEnv <: MuJoCoEnv
    data::MuJoCo.Data
    last_torso_x::Float64
    ctrl_squared::Float64
end

function RodentEnv()
    modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"
    model = MuJoCo.load_model(modelPath)
    RodentEnv(model)
end

function RodentEnv(model::MuJoCo.Model)
    data = MuJoCo.init_data(model)
    MuJoCo.reset!(model, data)
    last_torso_x = MuJoCo.body(data, "torso").com[1]
    RodentEnv(MuJoCo.init_data(model), last_torso_x, sum(data.ctrl .^ 2))
end

torso_x(env::RodentEnv) = MuJoCo.body(env.data, "torso").com[1]
torso_z(env::RodentEnv) = MuJoCo.body(env.data, "torso").com[3]
torso_speed_x(env::RodentEnv) = torso_x(env) - env.last_torso_x
state(model::MuJoCo.Model, env::RodentEnv) = MuJoCo.get_physics_state(model, env.data)

function act!(model, env::RodentEnv, action, params)
    env.last_torso_x = torso_x(env)
    env.ctrl_squared = sum(action .^ 2)
    env.data.ctrl .= clamp.(action, -1.0, 1.0)
    for _=1:params.n_physics_steps
        MuJoCo.step!(model, env.data)
    end
end

function reward(model, env::RodentEnv, params)
    velocity = torso_speed_x(env)
    forward_reward = params.forward_reward_weight * velocity
    healthy_reward = params.healthy_reward_weight * (!is_terminated(env, params))
    ctrl_reward = -params.ctrl_reward_weight * env.ctrl_squared
    return forward_reward + healthy_reward + ctrl_reward
end

is_terminated(env::RodentEnv, params) = torso_z(env) < params.min_torso_z

function reset!(model::MuJoCo.Model, env::RodentEnv)
    MuJoCo.reset!(model, env.data)
    env.last_torso_x = torso_x(env)
end