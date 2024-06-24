struct NamedBatchTuple{T <: NamedTuple}
    nt::T
end

function Base.view(nbt::NamedBatchTuple, inds...)
    NamedBatchTuple(map(x->view(x, pad_colons(inds, ndims(x))...), nbt.nt))
end

Base.keys(nbt::NamedBatchTuple) = keys(nbt.nt)
Base.pairs(nbt::NamedBatchTuple) = pairs(nbt.nt)
function Base.setindex!(nbt::NamedBatchTuple, val, inds...)
    for key in nbt.nt |> keys
        nbt.nt[key][pad_colons(inds, ndims(nbt.nt[key]))...] = val[key]
    end
end
Base.getproperty(nbt::NamedBatchTuple, key::Symbol) = key == :nt ? getfield(nbt, :nt) : nbt.nt[key]
Base.getindex(nbt::NamedBatchTuple, key::Symbol) = nbt.nt[key]

function gpu_zeros_from_template(template::NamedTuple, batch_size)
    NamedBatchTuple(map(x->CUDA.zeros(size(x)..., batch_size...), template))
end

function cpu_zeros_from_template(template::NamedTuple, batch_size)
    NamedBatchTuple(map(x->zeros(eltype(x), size(x)..., batch_size...), template))
end

pad_colons(inds, ndims) = (ntuple(_->:, Val(ndims-length(inds)))..., inds...)