mutable struct VariationalBottleneck{D<:Flux.Dense}
    linear_layer::D
    seed::UInt64
    cumul_kl::Float32
    sigma_scale::Float32
    kl_weight::Float32
end

Flux.@layer VariationalBottleneck
Flux.trainable(a::VariationalBottleneck) = (;linear_layer=a.linear_layer)

function VariationalBottleneck(layer_size::Pair, kl_weight; sigma_scale=3.0f0,
                               seed::UInt64=rand(UInt64), init=zeros32)
    input_size, output_size = layer_size
    linear_layer = Dense(input_size => 2*output_size, tanh, init=init)
    return VariationalBottleneck(linear_layer, seed, 0.0f0, Float32(sigma_scale), Float32(kl_weight))
end

function (layer::VariationalBottleneck)(input)
    linear_output = layer.linear_layer(input)
    mu, unscaled_logsigma = split_halfway(linear_output; dim=1)
    logsigma = layer.sigma_scale * unscaled_logsigma
    xsi = Flux.ignore() do
        layer.seed = (1664525 * layer.seed + 1013904223) % typeof(layer.seed)
        if mu isa CUDA.AnyCuArray
            CUDA.seed!(layer.seed)
            return CUDA.randn(size(mu)...)
        else
            Random.seed!(layer.seed)
            return randn(eltype(mu), size(mu)...)
        end
    end
    latent = mu .+ exp.(logsigma) .* xsi
    #TODO: check that this works with autodiff correctly
    layer.cumul_kl += 0.5f0 * sum(-2 .* logsigma .- 1 .+ exp.(2 .* logsigma) .+ mu.^2) / length(mu)
    return latent
end

function regularization_loss(layer::VariationalBottleneck)
    return layer.kl_weight * layer.cumul_kl
end

checkpoint_latent_state(layer::VariationalBottleneck) = (; seed = layer.seed)
function restore_latent_state!(layer::VariationalBottleneck, latent_state::NamedTuple{(:seed,), Tuple{UInt64}})
    layer.seed = latent_state.seed
    #Resetting cumul_kl here is very hacky but should work since restore_latent_state!
    #is called before each miniepoch. This function should maybe be renamed `prepare_epoch!`
    #or something?
    layer.cumul_kl = 0.0f0
end
