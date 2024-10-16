struct ComponentTensor{A <: AbstractArray, I <: NamedTuple}
    data::A
    index::I
end

function ComponentTensor(nt::NamedTuple)
    counter = Ref(1)
    function computeRanges(a)
        old_ind = counter[]
        counter[] += length(a)
        return old_ind:(counter[]-1)
    end
    computeRanges(nt::NamedTuple) = map(computeRanges, nt)
    ranges = computeRanges(nt)
    firstArray(nt::NamedTuple) = firstArray(first(nt))
    firstArray(a::AbstractArray) = a
    firstArray(a::Number) = nothing
    if firstArray(nt) isa CUDA.AnyCuArray
        data = CUDA.zeros(counter[]-1)
    else
        data = zeros(counter[]-1)
    end
    function copyData(range::NamedTuple, arrays::NamedTuple)
        for (key, value) in pairs(arrays)
            if value isa Number
                data[range[key]] .= value
            elseif value isa AbstractArray
                data[range[key]] .= reshape(value, :)
            else
                copyData(range[key], arrays[key])
            end
        end
    end
    copyData(ranges, nt)
    return ComponentTensor(data, ranges)
end

#function ComponentTensor(arrays::NamedTuple)
#    data = cat(values(arrays)...; dims=1)
#    sizes = [size(a, 1) for a=values(arrays)]
#    cmsm = cumsum(sizes)
#    start = [1; cmsm[1:end-1] .+ 1]
#    stop = cmsm
#    ks = keys(arrays)
#    #::NamedTuple{ks, NTuple{length(arrays), Tuple{UnitRange{Int64}, Tuple{Int64}}}}
#    indicies = NamedTuple{ks}(
#        (start[i]:stop[i], (sizes[i],)) for i=1:length(arrays)
#    )
#    ComponentTensor(data, indicies)
#end
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
    array(ct)[inds...] = array(value)
end

#function Base.setindex!(ct::ComponentTensor, value::ComponentTensor, key::Symbol, inds...)
#    data(ct)[range(ct, key), inds...] = data(value)
#end

function Base.setindex!(ct::ComponentTensor, value::NamedTuple, ind1::Colon, inds...)
    @assert keys(index(ct)) == keys(value)
    t(v::NamedTuple) = v
    t(v::AbstractArray) = reshape(v, :)
    t(v::Number) = SVector(v)
    for (k, v) in pairs(value)
        ct[k, inds...] = t(v)
    end
end

#function genFcnTest(nt::NamedTuple)
#    #@assert keys(index(ct)) == keys(nt)
#    lines = Expr[]
#    rec(nt::NamedTuple, prefix) = map(k->rec(nt[k], :($prefix.$k)), keys(nt))
#    rec(k, prefix) = push!(lines, :((ct).$prefix = nt.$prefix))#
#    map(k->rec(nt[k], k), keys(nt))
#    Expr(:block, lines...)
#end


function Base.setindex!(ct::ComponentTensor, value::NamedTuple, key::Symbol, inds...)
    @assert keys(index(ct)[key]) == keys(value)
    view(ct, key)[:, inds...] = value
end

function BatchComponentTensor(template::ComponentTensor, batch_dims...; array_fcn=zeros)
    ComponentTensor(array_fcn(size(template)..., batch_dims...), index(template))
end

function genFcnTest(nt::NamedTuple)
    lines = Expr[]
    
    # Recursive function to handle nested NamedTuples
    function rec(nt, prefix)
        for k in keys(nt)
            new_prefix = Expr(:., prefix, k)  # Construct valid field access
            if nt[k] isa NamedTuple
                rec(nt[k], new_prefix)
            else
                push!(lines, :(ct.$new_prefix = nt.$new_prefix))
            end
        end
    end
    
    rec(nt, :nt)
    return Expr(:block, lines...)  # Return a block of expressions
end

#function computeRange(ct::ComponentTensor, indkeys::Vector{Symbol})
#    for i = 2:length(indkeys)
#        @assert index(ct)[indkeys[i]][1].start == index(ct)[indkeys[i-1]][1].stop + 1
#    end
#    start = index(ct)[first(indkeys)][1].start
#    stop = index(ct)[last(indkeys)][1].stop
#    return start:stop
#end

#function Base.getindex(ct::ComponentTensor, keys::Vector{Symbol}, inds...)
#    data(ct)[computeRange(ct, keys), inds...]
#end
#Base.getindex(ct::ComponentTensor, key::Vector{Symbol}) = ct[key, ntuple(_->:, ndims(ct)-1)...]

#function Base.view(ct::ComponentTensor, indkeys::Vector{Symbol}, inds...)
#    view(data(ct), computeRange(ct, indkeys), inds...)
#end
#function Base.view(ct::ComponentTensor, indkeys::Vector{Symbol})
#    if ndims(ct) > 1
#        return view(ct, indkeys, ntuple(_->:, ndims(ct)-1)...)
#    else
#        return view(data(ct), computeRange(ct, indkeys))
#    end
#end

function Base.view(ct::ComponentTensor, key::Symbol)
    getproperty(ct, key)
    #if ndims(ct) > 1
    #    return view(ct, key, ntuple(_->:, ndims(ct)-1)...)
    #else
    #    return view(data(ct), range(ct, key))
    #end
end

Base.setproperty!(ct::ComponentTensor, key::Symbol) = view(ct, key, ntuple(_->:, ndims(ct)-1)...)

