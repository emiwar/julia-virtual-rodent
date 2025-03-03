struct RecordableLayer{F, L, K <: Tuple}
    rec::F
    layer::L
    key::K
end

Flux.trainable(rl::RecordableLayer) = Flux.trainable(rl.layer)

recordable(rec, layer::Union{Flux.Dense, Flux.Recur}, key::Tuple) = RecordableLayer(rec, layer, key)

function recordable(rec, chain::Flux.Chain, key::Tuple)
    Flux.Chain(NamedTuple(map(pairs(chain)) do (name, layer)
        name => recordable(rec, layer, (key..., name))
    end)...)
end


reset_layer!(rl::RecordableLayer, reset_mask) = reset_layer!(rl.layer, reset_mask)
checkpoint_latent_state(rl::RecordableLayer) = checkpoint_latent_state(rl.layer)
restore_latent_state!(rl::RecordableLayer, latent_state) = restore_latent_state!(rl.layer, latent_state)
regularization_loss(rl::RecordableLayer) = regularization_loss(rl.layer)
function Base.show(io::IO, l::RecordableLayer)
    print(io, "Recording of ")
    print(io, l.layer)
end

#Specializations
function (rl::RecordableLayer{F, RL})(x) where {F <: Function, L <: Flux.LSTMCell, RL <: Flux.Recur{L}}
    y = rl.layer(x)
    rl.rec((rl.key..., :h), rl.layer.state[1])
    rl.rec((rl.key..., :c), rl.layer.state[2])
    return y
end

function (rl::RecordableLayer{F, D})(x) where {F <: Function, D <: Flux.Dense}
    y = rl.layer(x)
    rl.rec(rl.key, y)
    return y
end

function (rl::RecordableLayer{F, VB})(x) where {F <: Function, VB <: VariationalBottleneck}
    #A bit clumsy to compute mu and sigma twice, but should be fine
    linear_output = rl.layer.linear_layer(input)
    mu, unscaled_logsigma = split_halfway(linear_output; dim=1)
    sigma = exp.(rl.layer.sigma_scale .* unscaled_logsigma)
    rl.rec((rl.key..., :mu), mu)
    rl.rec((rl.key..., :sigma), sigma)

    y = rl.layer(x)
    rl.rec((rl.key..., :latent), y)
    return y
end

function (rl::RecordableLayer{F, L})(x) where {F <: Function, L}
    @error "RecordableLayer has only been implemented for Dense, LSTM and
            VariationalBottleneck layers. Please implement for layer type $L."
end

function recordable(rec, actor_critic::AbstractEncDec)
    return EncDec(recordable(rec, actor_critic.encoder, (:encoder,)),
                  recordable(rec, actor_critic.decoder, (:decoder,)),
                  recordable(rec, actor_critic.critic, (:critic,)),
                  actor_critic.action_sampler)
end
