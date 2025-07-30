@concrete struct NerveNet
    input_layers<:NamedTuple
    afferents<:NamedTuple
    efferents<:NamedTuple
    motor_layers<:NamedTuple
    critics<:NamedTuple
    action_sampler
end

@Flux.layer NerveNet

function NerveNet(template_state::ComponentVector, template_action::ComponentVector;
                  sigma_min=0.01, sigma_max=0.5, actor_hidden_size=32, critic_hidden_size=nothing)
    input_sizes = map(k->k=>size(getproperty(template_state,k), 1), keys(template_state)) |> NamedTuple
    output_sizes = map(k->k=>size(getproperty(template_action,k), 1), keys(template_action)) |> NamedTuple
    input_layers = map(s->Dense(s=>actor_hidden_size, tanh), input_sizes)
    afferents    = map(_->Dense(actor_hidden_size=>actor_hidden_size, tanh), input_sizes)
    efferents    = map(_->Dense(actor_hidden_size=>actor_hidden_size, tanh), input_sizes)
    motor_layers = map(s->Dense(actor_hidden_size=>2*s, tanh, init=zeros32), output_sizes)
    critics      = map(_->Dense(actor_hidden_size=>1, init=zeros32), input_sizes)
    action_sampler = ModularActionSampler(;sigma_min, sigma_max)
    return NerveNet(input_layers, afferents, efferents, motor_layers, critics, action_sampler)
end

NerveNet(template_state::NamedTuple; kwargs...) = NerveNet(ComponentVector(template_state); kwargs...)

function actor(nn::NerveNet, states::ComponentArray, reset_index, action=nothing)
    #Input pass
    x_hand_L = nn.input_layers.hand_L(states.hand_L |> autodiff_clean)
    x_arm_L = nn.input_layers.arm_L(states.arm_L |> autodiff_clean) 
    x_hand_R = nn.input_layers.hand_R(states.hand_R |> autodiff_clean)
    x_arm_R = nn.input_layers.arm_R(states.arm_R |> autodiff_clean)
    x_foot_L = nn.input_layers.foot_L(states.foot_L |> autodiff_clean)
    x_leg_L = nn.input_layers.leg_L(states.leg_L |> autodiff_clean) + nn.afferents.foot_L(x_foot_L)
    x_foot_R = nn.input_layers.foot_R(states.foot_R |> autodiff_clean)
    x_leg_R = nn.input_layers.leg_R(states.leg_R |> autodiff_clean) + nn.afferents.foot_R(x_foot_R)
    x_torso = nn.input_layers.torso(states.torso |> autodiff_clean)
    x_head = nn.input_layers.head(states.head |> autodiff_clean)

    #Afferents
    x_arm_L = x_arm_L + nn.afferents.hand_L(x_hand_L)
    x_arm_R = x_arm_R + nn.afferents.hand_R(x_hand_R)
    x_foot_L = x_foot_L + nn.afferents.foot_L(x_foot_L)
    x_foot_R = x_foot_R + nn.afferents.foot_R(x_foot_R)

    #Combine
    x_root = nn.afferents.arm_L(x_arm_L) + nn.afferents.arm_R(x_arm_R) +
             nn.afferents.leg_L(x_leg_L) + nn.afferents.leg_R(x_leg_R) +
             nn.afferents.torso(x_torso) + nn.afferents.head(x_head)

    #Efferents
    x_head = x_head + nn.efferents.head(x_root)
    x_torso = x_torso + nn.efferents.torso(x_root)
    x_leg_R = x_leg_R + nn.efferents.leg_R(x_root)
    x_foot_R = x_foot_R + nn.efferents.foot_R(x_leg_R)
    x_leg_L = x_leg_L + nn.efferents.leg_L(x_root)
    x_foot_L = x_foot_L + nn.efferents.foot_L(x_leg_L)
    x_arm_R = x_arm_R + nn.efferents.arm_R(x_root)
    x_hand_R = x_hand_R + nn.efferents.hand_R(x_arm_R)
    x_arm_L = x_arm_L + nn.efferents.arm_L(x_root)
    x_hand_L = x_hand_L + nn.efferents.hand_L(x_arm_L)
    
    nn.action_sampler((
        hand_L = nn.motor_layers.hand_L(x_hand_L),
        arm_L = nn.motor_layers.arm_L(x_arm_L),
        hand_R = nn.motor_layers.hand_R(x_hand_R),
        arm_R = nn.motor_layers.arm_R(x_arm_R),
        foot_L = nn.motor_layers.foot_L(x_foot_L),
        leg_L = nn.motor_layers.leg_L(x_leg_L),
        foot_R = nn.motor_layers.foot_R(x_foot_R),
        leg_R = nn.motor_layers.leg_R(x_leg_R),
        torso = nn.motor_layers.torso(x_torso),
        head = nn.motor_layers.head(x_head),
    ), action)
end

function critic(nn::NerveNet, states::ComponentArray)
    #Input pass
    x_hand_L = nn.input_layers.hand_L(states.hand_L |> autodiff_clean)
    x_arm_L = nn.input_layers.arm_L(states.arm_L |> autodiff_clean) 
    x_hand_R = nn.input_layers.hand_R(states.hand_R |> autodiff_clean)
    x_arm_R = nn.input_layers.arm_R(states.arm_R |> autodiff_clean)
    x_foot_L = nn.input_layers.foot_L(states.foot_L |> autodiff_clean)
    x_leg_L = nn.input_layers.leg_L(states.leg_L |> autodiff_clean) + nn.afferents.foot_L(x_foot_L)
    x_foot_R = nn.input_layers.foot_R(states.foot_R |> autodiff_clean)
    x_leg_R = nn.input_layers.leg_R(states.leg_R |> autodiff_clean) + nn.afferents.foot_R(x_foot_R)
    x_torso = nn.input_layers.torso(states.torso |> autodiff_clean)
    x_head = nn.input_layers.head(states.head |> autodiff_clean)

    #Afferents
    x_arm_L = x_arm_L + nn.afferents.hand_L(x_hand_L)
    x_arm_R = x_arm_R + nn.afferents.hand_R(x_hand_R)
    x_foot_L = x_foot_L + nn.afferents.foot_L(x_foot_L)
    x_foot_R = x_foot_R + nn.afferents.foot_R(x_foot_R)

    #Combine
    x_root = nn.afferents.arm_L(x_arm_L) + nn.afferents.arm_R(x_arm_R) +
             nn.afferents.leg_L(x_leg_L) + nn.afferents.leg_R(x_leg_R) +
             nn.afferents.torso(x_torso) + nn.afferents.head(x_head)

    #Efferents
    x_head = x_head + nn.efferents.head(x_root)
    x_torso = x_torso + nn.efferents.torso(x_root)
    x_leg_R = x_leg_R + nn.efferents.leg_R(x_root)
    x_foot_R = x_foot_R + nn.efferents.foot_R(x_leg_R)
    x_leg_L = x_leg_L + nn.efferents.leg_L(x_root)
    x_foot_L = x_foot_L + nn.efferents.foot_L(x_leg_L)
    x_arm_R = x_arm_R + nn.efferents.arm_R(x_root)
    x_hand_R = x_hand_R + nn.efferents.hand_R(x_arm_R)
    x_arm_L = x_arm_L + nn.efferents.arm_L(x_root)
    x_hand_L = x_hand_L + nn.efferents.hand_L(x_arm_L)
    
    (
        hand_L = nn.critics.hand_L(x_hand_L) * 2f1,
        arm_L = nn.critics.arm_L(x_arm_L) * 2f1,
        hand_R = nn.critics.hand_R(x_hand_R) * 2f1,
        arm_R = nn.critics.arm_R(x_arm_R) * 2f1,
        foot_L = nn.critics.foot_L(x_foot_L) * 2f1,
        leg_L = nn.critics.leg_L(x_leg_L) * 2f1,
        foot_R = nn.critics.foot_R(x_foot_R) * 2f1,
        leg_R = nn.critics.leg_R(x_leg_R) * 2f1,
        torso = nn.critics.torso(x_torso) * 2f1,
        head = nn.critics.head(x_head) * 2f1,
    )
end

autodiff_clean(x) = Flux.ignore(()->copy(getdata(x)))

#Qs:
# * Scale critics?
# * Should actor and critic be combined?
# * Local recurrence within each node?
# * Exploration sigma propto frac of max value, rather than to surprise?
#   - Alternatively, relative to value of each state within each collected batch? 
# * Add root reward term