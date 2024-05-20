include("rodent_env.jl")
modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"
m = MuJoCo.load_model(modelPath)
env = RodentEnv(m)
RLBase.test_interfaces!(env)
RLBase.test_runnable!(env)

for i=1:1000
    act!(env, rand(action_space(env)))
    if (is_terminated(env))
        reset!(env)
    end
end

rews = RLBase.run(
    RandomPolicy(),
    RodentEnv(m),
    StopAfterNEpisodes(3),
    RewardsPerEpisode(),
    ResetAfterNSteps(5000)
)

function collect_rewards(env)
    rewards = Float64[]
    for i=1:1000
        act!(env, rand(action_space(env)))
        push!(rewards, reward(env))
        if (is_terminated(env))
            reset!(env)
        end
    end
    return rewards
end

