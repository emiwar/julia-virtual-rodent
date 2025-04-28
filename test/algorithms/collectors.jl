import TOML
import Dates
include("../../src/utils/parse_config.jl")
include("../../src/utils/component_tensor.jl")
include("../../src/utils/profiler.jl"); using .Timers
include("../../src/environments/environments.jl")
include("../../src/networks/networks.jl")
include("../../src/algorithms/algorithms.jl")

import Flux

params = parse_config(["configs/imitation_sota.toml"])

walker = Environments.Rodent(;params.physics...)
reward_spec = Environments.EqualRewardWeights(;params.reward...)
target = Environments.load_imitation_target(walker)
template_env = Environments.ImitationEnv(walker, reward_spec, target; params.imitation...)
env = Environments.MultithreadEnv(template_env, 512)

networks = Networks.EncDec(template_env, params)
collector = Algorithms.CuCollector(env, networks, params.rollout.n_steps_per_epoch)

@time Algorithms.collect_epoch!(collector, networks |> Flux.gpu)

