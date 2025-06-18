@concrete struct FootPosNet
    actors<:NamedTuple
    critics<:NamedTuple
    action_sampler
end

@Flux.layer FootPosNet

function FootPosNet(template_state::ComponentVector, template_action::ComponentVector;
                    sigma_min=0.01, sigma_max=0.5, actor_hidden_size=64, critic_hidden_size=64)
    input_sizes = map(k->k=>size(getproperty(template_state,k), 1), keys(template_state)) |> NamedTuple
    output_sizes = map(k->k=>size(getproperty(template_action,k), 1), keys(template_action)) |> NamedTuple
    actors = map(input_sizes, output_sizes) do input_size, output_size
        Chain(Dense(input_size=>actor_hidden_size, tanh),
              Dense(actor_hidden_size=>2*output_size, tanh, init=zeros32))
    end
    critics = map(input_sizes) do input_size
        Chain(Dense(input_size=>critic_hidden_size, tanh), Dense(critic_hidden_size=>1, init=zeros32))
    end
    action_sampler = ModularActionSampler(;sigma_min, sigma_max)
    return FootPosNet(actors, critics, action_sampler)
end

FootPosNet(template_state::NamedTuple; kwargs...) = FootPosNet(ComponentVector(template_state); kwargs...)

function actor(pc::FootPosNet, states::ComponentArray, reset_index, action=nothing)
    pc.action_sampler((
        torso = pc.actors.torso(states.torso |> autodiff_clean),
        arm_L = pc.actors.arm_L(states.arm_L |> autodiff_clean),
        arm_R = pc.actors.arm_R(states.arm_R |> autodiff_clean),
        leg_L = pc.actors.leg_L(states.leg_L |> autodiff_clean),
        leg_R = pc.actors.leg_R(states.leg_R |> autodiff_clean),
        head = pc.actors.head(states.head |> autodiff_clean),
    ), action)
end

function critic(pc::FootPosNet, states::ComponentArray)
    (
        torso = pc.critics.torso(states.torso |> autodiff_clean) .* 5f1,
        arm_L = pc.critics.arm_L(states.arm_L |> autodiff_clean) .* 2f1,
        arm_R = pc.critics.arm_R(states.arm_R |> autodiff_clean) .* 2f1,
        leg_L = pc.critics.leg_L(states.leg_L |> autodiff_clean) .* 2f1,
        leg_R = pc.critics.leg_R(states.leg_R |> autodiff_clean) .* 2f1,
        head = pc.critics.head(states.head |> autodiff_clean) .* 5f1,
    )
end

autodiff_clean(x) = Flux.ignore(()->copy(getdata(x)))
