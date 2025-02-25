randn_like(arr::AbstractArray) = randn(size(arr)...)
randn_like(arr::CUDA.AnyCuArray) = CUDA.randn(size(arr)...)

#For latent network states during debugging
function Base.isapprox(a::NamedTuple, b::NamedTuple)
    keys(a) == keys(b) && all(k -> a[k] ≈ b[k], keys(a))
end

function Base.isapprox(a::Tuple, b::Tuple)
    eachindex(a) == eachindex(a) && all(i -> a[i] ≈ b[i], eachindex(a))
end

function split_halfway(a::AbstractArray; dim=1)
    n = size(a, dim) ÷ 2
    return selectdim(a, dim, 1:n), selectdim(a, dim, n+1:2n)
end

function create_layers(Kind, group_name, sizes::AbstractVector, args...; kwargs...)
    map(1:(length(sizes)-1)) do i
        Symbol("$(group_name)_$i") => Kind(sizes[i] => sizes[i+1], args...; kwargs...)
    end
end

"Reset the state of a layer to its initial value. For single instance (e.g. a single environment
during a rollout), reset_mask is a boolean. For batches, reset_mask is a vector of booleans,
indicating whether to reset to correpsonding network state. For stateless (feedforward) layers,
this is a no-op."
reset_layer!(layer, reset_mask) = nothing #Stateless layers
function reset_layer!(layer::Flux.Recur, #Stateful layers
                      reset_mask::Union{Bool, AbstractVector{Bool}})
    layer.state = map(layer.state, layer.cell.state0) do s, s0
        ifelse.(reset_mask', s0, s)
    end
end

"Apply `chain` to `input`, but optionally first resetting the stateful layers for the
instances indicated by `reset_mask` (typically due to the correpsonding environments 
having been reset)."
function step!(chain::Flux.Chain, input, reset_mask)
    @assert ndims(reset_mask) == ndims(input) - 1
    for layer in chain
        reset_layer!(layer, reset_mask)
    end
    return chain(input)
end

"Take multiple steps of `chain`. The `input` is a 3D array (input_size x n_envs x n_steps),
with the third dimension being time. The `reset_mask` is a 2D array (n_envs x n_steps)."
function rollout!(chain::Flux.Chain, input::AbstractArray{T, 3}, reset_mask::AbstractMatrix{Bool}) where T
    @assert ndims(reset_mask) == ndims(input) - 1
    cat(map(axes(input, 3)) do t
        step!(chain, view(input, :, :, t), view(reset_mask, :, t))
    end...; dims=3)
end

function rollout!(chain::Flux.Chain, input::AbstractMatrix, reset_mask::AbstractVector{Bool})
    return step!(chain, input, reset_mask)
end

function rollout!(chain::Flux.Chain, input::AbstractVector, reset_mask::Bool)
    return step!(chain, input, reset_mask)
end

#Q: Should just work to use Flux.state here instead?
#A: NO, because if there is more than one miniepoch, we'd also reset the weights
#and biases between each miniepoch, which is not what we want.
checkpoint_latent_state(chain::Flux.Chain) = map(checkpoint_latent_state, chain.layers)
checkpoint_latent_state(layer::Flux.Recur) = layer.state
function checkpoint_latent_state(layer::T) where T
    if hasfield(T, :state) || hasfield(T, :seed)
        @warn "Looks like $T might be stateful, but no `checkpoint_latent_state` defined."
    end
    NamedTuple()
end
restore_latent_state!(chain::Flux.Chain, latent_state) = map(restore_latent_state!, chain.layers, latent_state)
restore_latent_state!(layer::Flux.Recur, latent_state::Tuple) = (layer.state = latent_state)
restore_latent_state!(layer::Any, latent_state::NamedTuple{()}) = nothing

regularization_loss(chain::Flux.Chain) = sum(regularization_loss.(chain.layers))
regularization_loss(layer::Any) = 0.0f0
