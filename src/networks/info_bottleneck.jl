mutable struct InfoBottleneck{D<:Flux.Dense}
    linear_layer::D
    seed::UInt64
    noise_scale::Float32
end

Flux.@layer InfoBottleneck
Flux.trainable(a::InfoBottleneck) = (;linear_layer=a.linear_layer)

function InfoBottleneck(layer_size::Pair; noise_scale=1.0f0,
                        seed::UInt64=rand(UInt64), init=zeros32)
    input_size, output_size = layer_size
    linear_layer = Dense(input_size => output_size, tanh, init=init)
    return InfoBottleneck(linear_layer, seed, Float32(noise_scale))
end

function (layer::InfoBottleneck)(input)
    mu = layer.linear_layer(input)
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
    latent = mu .+ layer.noise_scale .* xsi
    return latent
end

function regularization_loss(layer::InfoBottleneck)
    return 0.0f0
end

checkpoint_latent_state(layer::InfoBottleneck) = (; seed = layer.seed)
function restore_latent_state!(layer::InfoBottleneck, latent_state::NamedTuple{(:seed,), Tuple{UInt64}})
    layer.seed = latent_state.seed
end
