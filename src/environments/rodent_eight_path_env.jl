mutable struct RodentEightPathEnv <: RodentFollowEnv
    model::MuJoCo.Model
    data::MuJoCo.Data
    lifetime::Int64
    cumulative_reward::Float64
end

function RodentEightPathEnv()
    modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
    model = MuJoCo.load_model(modelPath)
    data = MuJoCo.init_data(model)
    env = RodentEightPathEnv(model, data, 0, 0.0)
    reset!(env)
    return env
end

function clone(env::RodentEightPathEnv)
    new_env = RodentEightPathEnv(
        env.model,
        MuJoCo.init_data(env.model),
        0, 0.0)
    reset!(new_env)
    return new_env
end

target_speed(env::RodentEightPathEnv) = 0.2
radius(env::RodentEightPathEnv) = 1.0

function get_future_targets(env::RodentEightPathEnv, params)
    factor = 2pi * radius(env) / target_speed(env) / env.model.opt.timestep / params.n_physics_steps
    t = (env.lifetime:(env.lifetime+params.imitation_steps_ahead)) ./ factor
    com = MuJoCo.body(env.data, "torso").com
    vcat(radius(env).*sin.(2.0 .* 2pi .* t') .- com[1],
         radius(env).*(cos.(2pi .* t') .- 1.0) .- com[2],
         zero(t'))
end

function get_target_vector(env::RodentEightPathEnv, params)
    factor = 2pi / target_speed(env) / env.model.opt.timestep / params.n_physics_steps
    t = (env.lifetime-1) / factor
    com = MuJoCo.body(env.data, "torso").com
    [radius(env)*sin(2.0*2pi*t) - com[1], radius(env)*(cos(2pi*t)-1.0) - com[2], 0.0]
end
