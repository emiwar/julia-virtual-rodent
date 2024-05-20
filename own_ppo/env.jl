import MuJoCo
modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"
m = MuJoCo.load_model(modelPath)

mutable struct EnvState
    data::MuJoCo.Data
    last_torso_x::Float64
    n_steps_taken::Int
end
function EnvState(model::MuJoCo.Model)
    data = MuJoCo.init_data(model)
    MuJoCo.reset!(model, data)
    last_torso_x = MuJoCo.body(data, "torso").com[1]
    EnvState(MuJoCo.init_data(model), last_torso_x, 0)
end

torso_x(env::EnvState) = MuJoCo.body(env.data, "torso").com[1]
torso_z(env::EnvState) = MuJoCo.body(env.data, "torso").com[3]
state(model::MuJoCo.Model, env::EnvState) = MuJoCo.get_physics_state(model, env.data)

function act!(model, env::EnvState, action, params)
    #println("Actin'")
    env.last_torso_x = torso_x(env)
    env.data.ctrl .= action
    for _=1:params.n_physics_steps
        MuJoCo.step!(model, env.data)
    end
    env.n_steps_taken += 1
    
end

function reward(model, env::EnvState, params)
    velocity = torso_x(env) - env.last_torso_x
    forward_reward = params.forward_reward_weight * velocity
    healthy_reward = params.healthy_reward_weight * (!is_terminated(env, params))
    ctrl_reward = params.ctrl_reward_weight * sum(env.data.ctrl .^ 2)
    return forward_reward + healthy_reward + ctrl_reward
end

is_terminated(env::EnvState, params) = torso_z(env) < params.min_torso_z

function reset!(model::MuJoCo.Model, env::EnvState)
    MuJoCo.reset!(model, env.data)
    env.last_torso_x = torso_x(env)
    env.n_steps_taken = 0
end