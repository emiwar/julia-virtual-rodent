import Plots
env = RodentImitationEnv()
reset!(env, params)
actor_critic = BSON.load(filename)[:actor_critic] |> Flux.gpu

function target_plot(targets; zoom=0.2)
    p = Plots.scatter(targets[1, :], targets[2, :], label=false,
                       xlim=(-zoom, zoom), ylim=(-zoom, zoom))
    Plots.plot!(p, targets[1, :], targets[2, :], label=false)
    return p
end

anim = @Plots.animate for t=1:2000
    env_state = state(env, params)
    actor_output = actor(actor_critic, ComponentTensor(CUDA.cu(data(env_state)), index(env_state)), params)
    action = clamp.(actor_output.action, -1.0, 1.0) |> Array
    act!(env, action, params)
    if is_terminated(env, params)
        reset!(env, params)
    end
    com_ind = get_com_ind(env)
    range = (com_ind + 1):(com_ind + params.imitation_steps_ahead)
    targets_absolute = view(env.com_targets, :, range)
    targets_relative = get_future_targets(env, params)
    com = MuJoCo.body(env.data, "torso").com
    xmat = reshape(MuJoCo.body(env.data, "torso").xmat, 3, 3)
    p1 = target_plot(targets_absolute)
    Plots.scatter!(p1, com[1:1], com[2:2], label=false)
    p2 = target_plot(targets_relative)
    p3 = target_plot(xmat*targets_relative)
    Plots.plot(p1, p2, p3)
end
Plots.gif(anim, "com_lookahead.mp4")
