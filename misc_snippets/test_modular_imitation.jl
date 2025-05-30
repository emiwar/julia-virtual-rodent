using BenchmarkTools
using ComponentArrays
import Flux
include("../src/utils/profiler.jl")
include("../src/environments/environments.jl")
include("../src/networks/networks.jl")
include("../src/algorithms/algorithms.jl")

walker = Environments.ModularRodent(min_torso_z=0.035, spawn_z_offset=0.01, n_physics_steps=5)

Environments.proprioception(walker)
env = Environments.ModularImitationEnv(walker; max_target_distance=0.1)
Environments.state(env)
Environments.reward(env)
Environments.status(env)
Environments.info(env)
a = Environments.null_action(env)
Environments.act!(env, a)
Environments.act!(env, ComponentVector(a))
Environments.reset!(env)

multienv = Environments.MultithreadEnv(env, 4)
Environments.state(multienv)

template_state  = ComponentVector(Environments.state(env))
template_action = ComponentVector(Environments.null_action(env))
net = Networks.ParallelControl(template_state, template_action)

Networks.actor(net, ComponentVector(template_state), false)

collector = Algorithms.CuCollector(multienv, net, 16)

actor_output = Networks.actor(Flux.gpu(net), collector.states[:, :, 1], nothing)
Networks.critic(Flux.gpu(net), collector.states[:, :, 1])

Algorithms.collect_epoch!(collector, Flux.gpu(net))


