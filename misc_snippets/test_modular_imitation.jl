using BenchmarkTools
using ComponentArrays
import Flux
include("../src/utils/profiler.jl")
include("../src/environments/environments.jl")
include("../src/networks/networks.jl")
include("../src/algorithms/algorithms.jl")

walker = Environments.ModularRodent(min_torso_z=0.035, spawn_z_offset=0.01, n_physics_steps=5)
env = Environments.ModularImitationEnv(walker; max_target_distance=0.1)

Environments.proprioception(walker)
Environments.status(env)
Environments.info(env)
a = Environments.null_action(env)
Environments.act!(env, a)
Environments.act!(env, ComponentVector(a))
Environments.reset!(env)
Environments.reward(env)

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


values = Networks.critic(Flux.gpu(net), collector.states)


S = view(collector.states, :, :, 1:16)
A = collector.actor_outputs.action
Flux.gradient(Flux.gpu(net)) do actor_critic
    actor_output = Networks.actor(actor_critic, S, false, A)
    LL = vcat(actor_output.loglikelihood...)
    sum(LL)#sum(actor_output.loglikelihood.hand_L)
end

params = (
    training = (
        gamma = 0.95,
        lambda = 0.95,
        clip_range = 0.2,
        learning_rate = 1e-4,
        loss_weight_actor = 1.0,
        loss_weight_critic = 0.5,
        loss_weight_entropy = -0.2,
        n_miniepochs = 1,
    ),
    rollout = (
        n_epochs = 10,
    )
)

Algorithms.mppo(collector, Flux.gpu(net), params)

