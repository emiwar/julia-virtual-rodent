@concrete struct ParallelControl
    actors<:NamedTuple
    critics<:NamedTuple
    action_sampler
end

@Flux.layer ParallelControl

function ParallelControl(template_state::ComponentVector, template_action::ComponentVector;
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
    return ParallelControl(actors, critics, action_sampler)
end

ParallelControl(template_state::NamedTuple; kwargs...) = ParallelControl(ComponentVector(template_state); kwargs...)

#function actor(pc::ParallelControl, states::NamedTuple)
#    split_states = NamedTuple(k=>getproperty(states, k) for k in keys(states))
#    map(pc.actors, keys(actor)) do (actor, key)
#        actor(getproperty(states, key)) |> pc.action_sampler
#    end
#   actor_outputs = ComponentArray((k=>pc.actors[k](getproperty(states, k)) for k in keys(pc.actors))...)
#   pc.action_sampler(actor_outputs)
#end

function actor(pc::ParallelControl, states::ComponentArray, reset_index, action=nothing)
    pc.action_sampler((
        hand_L = pc.actors.hand_L(states.hand_L |> autodiff_clean),
        arm_L = pc.actors.arm_L(states.arm_L |> autodiff_clean),
        hand_R = pc.actors.hand_R(states.hand_R |> autodiff_clean),
        arm_R = pc.actors.arm_R(states.arm_R |> autodiff_clean),
        foot_L = pc.actors.foot_L(states.foot_L |> autodiff_clean),
        leg_L = pc.actors.leg_L(states.leg_L |> autodiff_clean),
        foot_R = pc.actors.foot_R(states.foot_R |> autodiff_clean),
        leg_R = pc.actors.leg_R(states.leg_R |> autodiff_clean),
        torso = pc.actors.torso(states.torso |> autodiff_clean),
        head = pc.actors.head(states.head |> autodiff_clean),
    ), action)
end

function critic(pc::ParallelControl, states::ComponentArray)
    (
        hand_L = pc.critics.hand_L(states.hand_L |> autodiff_clean) .* 2f1,
        arm_L = pc.critics.arm_L(states.arm_L |> autodiff_clean) .* 2f1,
        hand_R = pc.critics.hand_R(states.hand_R |> autodiff_clean) .* 2f1,
        arm_R = pc.critics.arm_R(states.arm_R |> autodiff_clean) .* 2f1,
        foot_L = pc.critics.foot_L(states.foot_L |> autodiff_clean) .* 2f1,
        leg_L = pc.critics.leg_L(states.leg_L |> autodiff_clean) .* 2f1,
        foot_R = pc.critics.foot_R(states.foot_R |> autodiff_clean) .* 2f1,
        leg_R = pc.critics.leg_R(states.leg_R |> autodiff_clean) .* 2f1,
        torso = pc.critics.torso(states.torso |> autodiff_clean) .* 5f1,
        head = pc.critics.head(states.head |> autodiff_clean) .* 5f1,
    )
end

autodiff_clean(x) = Flux.ignore(()->copy(getdata(x)))