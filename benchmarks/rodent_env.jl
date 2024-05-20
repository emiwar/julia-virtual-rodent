using ReinforcementLearning
import MuJoCo

@kwdef mutable struct RodentEnv <: ReinforcementLearning.AbstractEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    n_physics_steps::Int = 5
    min_body_z::Float64 = 0.035
    forward_reward_weight::Float64 = 10.0
    healthy_reward_weight::Float64 = 1.0
    ctrl_reward_weight::Float64 = 0.1

    n_steps_taken::Int = 0
    last_torso_com_x::Float64 = NaN
end

function RodentEnv(model::MuJoCo.Model)
    env = RodentEnv(;model=model, data=MuJoCo.init_data(model))
    reset!(env)
    env
end

control_size(env::RodentEnv) = env.model.nu |> Int
obs_size(env::RodentEnv) = MuJoCo.mj_stateSize(env.model, MuJoCo.LibMuJoCo.mjSTATE_PHYSICS) |> Int
torso_com(env::RodentEnv) = MuJoCo.body(env.data, "torso").com

RLBase.action_space(env::RodentEnv) = (-1.0 .. 1.0) ^ control_size(env)
RLBase.state_space(env::RodentEnv) = (-Inf .. Inf) ^ obs_size(env)
RLBase.StateStyle(env::RodentEnv) = Observation{Vector{Float64}}()
RLBase.ChanceStyle(env::RodentEnv) = Deterministic()


RLBase.state(env::RodentEnv) = MuJoCo.get_physics_state(env.model, env.data)

RLBase.is_terminated(env::RodentEnv) = torso_com(env)[3] < env.min_body_z

function RLBase.reset!(env::RodentEnv)
    MuJoCo.reset!(env.model, env.data)
    env.last_torso_com_x = torso_com(env)[1]
    env.n_steps_taken = 0
end

function RLBase.act!(env::RodentEnv, action)
    #println("Actin'")
    env.last_torso_com_x = torso_com(env)[1]
    env.data.ctrl .= action
    for _=1:env.n_physics_steps
        MuJoCo.step!(env.model, env.data)
    end
    env.n_steps_taken += 1
    
end

function RLBase.reward(env::RodentEnv)
    velocity = torso_com(env)[1] - env.last_torso_com_x
    forward_reward = env.forward_reward_weight * velocity
    healthy_reward = env.healthy_reward_weight * (!is_terminated(env))
    ctrl_reward = env.ctrl_reward_weight * sum(env.data.ctrl .^ 2)
    return forward_reward + healthy_reward + ctrl_reward
end

RLBase.seed!(env::RodentEnv, seed) = nothing

Base.show(io::IO, ::MIME"text/plain", env::RodentEnv) = Base.show(io, string(env))

