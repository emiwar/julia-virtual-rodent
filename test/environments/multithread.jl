import TOML
import Dates
include("../../src/utils/parse_config.jl")
include("../../src/utils/component_tensor.jl")
include("../../src/environments/environments.jl")

params = parse_config(["configs/imitation_sota.toml"])

walker = Environments.Rodent(;params.physics...)
reward_spec = Environments.EqualRewardWeights(;params.reward...)
target = Environments.load_imitation_target(walker)
template_env = Environments.ImitationEnv(walker, reward_spec, target; params.imitation...)
env = Environments.MultithreadEnv(template_env, 512)

Environments.prepare_epoch!(env)
Environments.state(env)
Environments.status(env)
Environments.info(env)

@time Environments.act!(env, rand(walker.model.na, length(env.environments)))
Environments.reward(env)
Environments.state(env)
Environments.status(env)
Environments.info(env)
