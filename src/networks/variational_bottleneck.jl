mutable struct VariationalBottleneck{D<:Flux.Dense, A<:AbstractArray}
    linear_layer::D
    mu::A
    logsigma::A
    sigma_scale::Float32
    kl_weight::Float32
    seed::UInt64
end

function VariationalBottleneck(layer_size::Pair, kl_weight; sigma_scale=3.0f0, seed=rand(UInt64), init=zeros32)
    input_size, output_size = layer_size
    linear_layer = Dense(input_size => 2*output_size, tanh, init=init)
    mu = zeros(Float32, output_size, 0, 0)
    logsigma = zeros(Float32, output_size, 0, 0)
    return VariationalBottleneck(linear_layer, mu, logsigma, sigma_scale, kl_weight, seed)
end

Flux.@layer VariationalBottleneck

function (layer::VariationalBottleneck)(input)
    linear_output = layer.linear_layer(input)
    layer.mu, unscaled_logsigma = split_halfway(linear_output; dim=1)
    layer.logsigma = layer.sigma_scale * unscaled_logsigma
    xsi = Flux.ignore() do
        layer.seed = (1664525 * layer.seed + 1013904223) % typeof(layer.seed)
        if mu isa CUDA.AnyCuArray
            CUDA.seed!(layer.seed)
            return CUDA.randn(eltype(layer.mu), size(layer.mu)...)
        else
            Random.seed!(layer.seed)
            return randn(eltype(layer.mu), size(layer.mu)...)
        end
    end
    latent = layer.mu .+ exp.(layer.logsigma) .* xsi
    return latent
end

checkpoint_latent_state(layer::VariationalBottleneck) = (; seed = layer.seed)

function regularization_loss(layer::VariationalBottleneck)
    kl_div = 0.5*sum( -2 .* layer.logsigma .- 1 .+ exp.(2 .* layer.logsigma) .+ layer.mu.^2) / length(layer.mu)
    return layer.kl_weight * kl_div
end
