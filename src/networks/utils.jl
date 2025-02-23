randn_like(arr::AbstractArray) = randn(size(arr)...)
randn_like(arr::CUDA.AnyCuArray) = CUDA.randn(size(arr)...)

#For comparing LSTM states during debugging
function Base.isapprox(a::Vector{Tuple{T, T}}, b::Vector{Tuple{T, T}}) where {T}
    if length(a) != length(b) 
        return false
    end
    for (a_, b_) in zip(a, b)
        if !(a_[1] ≈ b_[1] && a_[2] ≈ b_[2])
            return false
        end
    end
    return true
end

function split_halfway(a::AbstractArray; dim=1)
    n = size(a, dim) ÷ 2
    return selectdim(a, dim, 1:n), selectdim(a, dim, n+1:2n)
end

function create_layers(Kind::Function, group_name, sizes::AbstractVector, args...; kwargs...)
    NamedTuple(map(1:(length(sizes)-1)) do i
        Symbol("$(group_name)_i") => Kind(sizes[i] => sizes[i+1], args...; kwargs...)
    end...)
end

"Reset the state of a layer to its initial value. For single instance (e.g. a single environment
during a rollout), reset_mask is a boolean. For batches, reset_mask is a vector of booleans,
indicating whether to reset to correpsonding network state. For stateless (feedforward) layers,
this is a no-op."
reset_layer(layer, reset_mask) = nothing #Stateless layers
function reset_layer(layer::Flux.Recur, reset_mask) #Stateful layers
    layer.state = map(layer.state, layer.cell.state0) do (s, s0)
        ifelse.(reset_mask', s0, s)
    end
end

"Apply `chain` to `input`, but optionally first resetting the stateful layers for the
instance indicated by `reset_mask` (typically due to the correpsonding environments have
reset)."
function step(chain::Flux.Chain, input::AbstractVecOrMat, reset_mask::AbstractVector)
    reset_layer.(chain, reset_mask)
    return chain(input)
end

"Take multiple steps of `chain`. The `input` is a 3D array, with the third dimension
being time. The `reset_mask` is a 2D array (n_envs x n_steps)."
function rollout(chain::Flux.Chain, input, reset_mask)
    @assert ndims(reset_mask) == ndims(input) - 1
    cat(map(axes(input, 3)) do t
        #TODO: additionally return state of statful layers when recording activations
        step(chain, view(input, :, :, t), view(reset_mask, :, t))
    end...; dims=3)
end

#Q: Should just work to use Flux.state here instead?
#A: NO, because if there is more than one miniepoch, we'd also reset the weights
#and biases between each miniepoch, which is not what we want.
checkpoint_latent_state(chain::Flux.Chain) = map(checkpoint_latent_state, chain)
checkpoint_latent_state(layer::Flux.Recur) = layer.state
function checkpoint_latent_state(layer::T) where T
    if hasfield(T, :state) || hasfield(T, :seed)
        @warn "Looks like $T might be stateful, but no checkpoint_latent_state method defined."
    end
    NamedTuple()
end
restore_latent_state!(chain::Flux.Chain, latent_state) = map(restore_latent_state!, chain, latent_state)
restore_latent_state!(layer::Flux.Recur, latent_state::Tuple) = (layer.state = latent_state)
restore_latent_state!(layer::Any, latent_state) = nothing
