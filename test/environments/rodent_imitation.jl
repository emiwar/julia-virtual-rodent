import TOML
import Dates
include("../../src/utils/parse_config.jl")
include("../../src/environments/environments.jl")

params = parse_config(["configs/imitation_sota.toml"])

walker = Environments.Rodent(;params.physics...)
reward_spec = Environments.EqualRewardWeights(;params.reward...)
target = Environments.load_imitation_target(walker)
env = Environments.ImitationEnv(walker, reward_spec, target; params.imitation...)

@time Environments.reset!(env)
@time Environments.state(env)
@time Environments.status(env)
@time Environments.info(env)
@time Environments.reward(env)
@time Environments.act!(env, rand(walker.model.na))

