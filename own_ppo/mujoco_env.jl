
import MuJoCo
using DomainSets

abstract type MuJoCoEnv end

mutable struct RodentEnv <: MuJoCoEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    last_torso_x::Float64
    ctrl_squared::Float64
    lifetime::Int64
    cumulative_reward::Float64
    #torso::MuJoCo.Wrappers.NamedAccess.DataBody
end

function RodentEnv()
    modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"
    model = MuJoCo.load_model(modelPath)
    RodentEnv(model)
end

function RodentEnv(model::MuJoCo.Model)
    data = MuJoCo.init_data(model)
    MuJoCo.reset!(model, data)
    torso = MuJoCo.body(data, "torso")
    env = RodentEnv(model, data, 0.0, 0.0, 0, 0.0)
    env.ctrl_squared = sum(env.data.ctrl .^ 2)
    reset!(env)
    return env
end

#Utils
torso_x(env::RodentEnv) = MuJoCo.body(env.data, "torso").com[1]
torso_y(env::RodentEnv) = MuJoCo.body(env.data, "torso").com[2]
torso_z(env::RodentEnv) = MuJoCo.body(env.data, "torso").com[3]
torso_speed_x(env::RodentEnv) = (torso_x(env) - env.last_torso_x) / env.model.opt.timestep

#Space declarations
function action_space(env::RodentEnv)
    (;torques=(-1.0 .. 1.0) ^ Int(env.model.nu))
end

function state_space(env::RodentEnv)
    (qpos=(-Inf .. Inf) ^ Int(env.model.nq),
     qvel=(-Inf .. Inf) ^ Int(env.model.nv),
     act=(-Inf .. Inf) ^ Int(env.model.na))
end

function info_space(env::RodentEnv)
    (torso_x=(-Inf .. Inf),
     torso_y=(-Inf .. Inf),
     torso_z=(-Inf .. Inf),
     torso_speed_x=(-Inf .. Inf),
     lifetime=(0 .. typemax(Int)),
     cumulative_reward=(0 .. Inf))
end

#Read-outs
function state(env::RodentEnv, params)
    (qpos=env.data.qpos,
     qvel=env.data.qvel,
     act=env.data.act)
end

function reward(env::RodentEnv, params)
    forward_reward = params.forward_reward_weight * torso_speed_x(env)
    healthy_reward = params.healthy_reward_weight * (!is_terminated(env, params))
    ctrl_reward = -params.ctrl_reward_weight * env.ctrl_squared
    return forward_reward + healthy_reward + ctrl_reward
end

function is_terminated(env::RodentEnv, params)
    torso_z(env) < params.min_torso_z
end

function info(env::RodentEnv)
    (torso_x=torso_x(env),
     torso_y=torso_y(env),
     torso_z=torso_z(env),
     torso_speed_x=torso_speed_x(env),
     lifetime=env.lifetime,
     cumulative_reward=env.cumulative_reward)
end

#Actions
function act!(env::RodentEnv, action, params)
    env.last_torso_x = torso_x(env)
    env.ctrl_squared = sum(action.torques .^ 2)
    env.data.ctrl .= clamp.(action.torques, -1.0, 1.0)
    for _=1:params.n_physics_steps
        MuJoCo.step!(env.model, env.data)
    end
    env.lifetime += 1
    env.cumulative_reward += reward(env, params)
end

function reset!(env::RodentEnv)
    MuJoCo.reset!(env.model, env.data)
    env.last_torso_x = torso_x(env)
    env.lifetime = 0
    env.cumulative_reward = 0.0
end

