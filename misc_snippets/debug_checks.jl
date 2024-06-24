actor_critic = ActorCritic(test_env, params) |> Flux.gpu
batch = collect_batch(envs, actor_critic, params);

dummy_action = (;torques=CUDA.fill(0.23f0, 38, 512, 9))

res = actor(actor_critic, batch.states, params, dummy_action);
res.mu
opt_state = Flux.setup(Flux.Adam(), actor_critic)

gradients = Flux.gradient(actor_critic) do actor_critic
    actor_output = actor(actor_critic, batch.states, params, dummy_action)
    return sum(actor_output.loglikelihood) /  length(actor_output.loglikelihood)
end

Flux.update!(opt_state, actor_critic, gradients[1])