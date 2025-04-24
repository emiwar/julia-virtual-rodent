module ComponentTensors
export ComponentTensor, BatchComponentTensor, data, index, array

struct ComponentTensor{A <: AbstractArray, I <: NamedTuple}
    data::A
    index::I
end

function ComponentTensor(nt::NamedTuple)
    counter = Ref(1)
    function computeRanges(a)
        old_ind = counter[]
        counter[] += size(a, 1)
        return old_ind:(counter[]-1)
    end
    computeRanges(nt::NamedTuple) = map(computeRanges, nt)
    ranges = computeRanges(nt)
    firstArray(nt::NamedTuple) = firstArray(first(nt))
    firstArray(a::AbstractArray) = a
    firstArray(a::Number) = nothing
    batch_size = isnothing(firstArray(nt)) ? () : size(firstArray(nt))[2:end]
    batch_dims = ntuple(_->:, length(batch_size))
    if isdefined(Main, :CUDA) && firstArray(nt) isa CUDA.AnyCuArray
        data = CUDA.zeros(counter[]-1, batch_size...)
    else
        data = zeros(counter[]-1, batch_size...)
    end
    function copyData(range::NamedTuple, arrays::NamedTuple)
        for (key, value) in pairs(arrays)
            if value isa Number
                data[range[key], batch_dims...] .= value
            elseif value isa AbstractArray
                data[range[key], batch_dims...] .= value#reshape(value, :)
            else
                copyData(range[key], arrays[key])
            end
        end
    end
    copyData(ranges, nt)
    return ComponentTensor(data, ranges)
end

ComponentTensor(;arrays...) = ComponentTensor(NamedTuple(arrays))
data(ct::ComponentTensor) = getfield(ct, :data)
index(ct::ComponentTensor) = getfield(ct, :index)
Base.keys(ct::ComponentTensor) = keys(index(ct))
array(ct::ComponentTensor) = view(data(ct), rangestart(index(ct)):rangestop(index(ct)), ntuple(_->:, ndims(ct)-1)...)
rangestart(nt::NamedTuple) = rangestart(first(nt))
rangestart(nt::UnitRange) = first(nt)
rangestop(nt::NamedTuple) = rangestop(last(nt))
rangestop(nt::UnitRange) = last(nt)
range(ct::ComponentTensor, key::Symbol) = rangestart(index(ct)[key]):rangestop(index(ct)[key])

function Base.size(ct::ComponentTensor)
    (rangestop(index(ct)) - rangestart(index(ct)) + 1, size(data(ct))[2:end]...)
end
Base.length(ct::ComponentTensor) = prod(size(ct))
Base.eltype(ct::ComponentTensor) = eltype(data(ct))
Base.ndims(ct::ComponentTensor) = ndims(data(ct))

function Base.getindex(ct::ComponentTensor, key::Symbol, inds...)
    data(ct)[range(ct, key), inds...]
end
function Base.getindex(ct::ComponentTensor, key::Symbol)
    if ndims(ct) > 1
        ct[key, ntuple(_->:, ndims(ct)-1)...]
    else
        return data(ct)[range(ct, key)]
    end
end

function Base.getproperty(ct::ComponentTensor, key::Symbol)
    if index(ct)[key] isa NamedTuple
        return ComponentTensor(data(ct), index(ct)[key])
    else
        return ComponentTensor(data(ct), NamedTuple((key=>index(ct)[key],)))
    end
end
function Base.view(ct::ComponentTensor, ind1::Colon, inds...)
    ComponentTensor(view(data(ct), ind1, inds...), index(ct))
end
function Base.view(ct::ComponentTensor, key::Symbol, inds...)
    view(data(ct), range(ct, key), inds...)
end

Base.setindex!(ct::ComponentTensor, value, inds...) = array(ct)[inds...] = value

function Base.setindex!(ct::ComponentTensor, value, key::Symbol, inds...)
    data(ct)[range(ct, key), inds...] = value
end

function Base.setindex!(ct::ComponentTensor, value::ComponentTensor, inds...)
    @assert index(ct) == index(value)
    data(ct)[inds...] = data(value)
    #array(ct)[inds...] = array(value)
end

function BatchComponentTensor(template::ComponentTensor, batch_dims...; array_fcn=zeros)
    ComponentTensor(array_fcn(size(template)..., batch_dims...), index(template))
end

@generated function Base.setindex!(ct::ComponentTensor, nt::NamedTuple, ind1::Colon, inds...)
    #Hacky meta-programming to avoid heap-allocations.
    #TODO: recursively check keys(ct) == keys(nt)
    lines = Expr[]
    
    # Recursive function to handle nested NamedTuples
    function rec(nt, lvalue, rvalue)
        keys, types = nt.parameters
        for (k, T) in zip(keys, types.parameters)
            new_lvalue = :(getproperty($lvalue, $(QuoteNode(k))))
            new_rvalue = :(getproperty($rvalue, $(QuoteNode(k))))
            if T <: NamedTuple
                rec(T, new_lvalue, new_rvalue)
            else
                push!(lines, quote 
                    lsize = size(view(data(ct), $new_lvalue, inds...))
                    rsize = size($new_rvalue)
                    ks = $(QuoteNode(k))
                    if lsize != rsize && !(lsize==(1,) && rsize==())
                        error("Size of key $ks does not match ($lsize vs $rsize)")
                    end
                    data(ct)[$new_lvalue, inds...] .= $new_rvalue
                end)
            end
        end
    end
    rec(nt, :(index(ct)), :nt)
    push!(lines, quote nothing end)
    return Expr(:block, lines...)  # Return a block of expressions
end

function Base.view(ct::ComponentTensor, key::Symbol)
    getproperty(ct, key)
end

end