struct ComponentTensor{A <: AbstractArray, I <: NamedTuple}
    data::A
    index::I
end

function ComponentTensor(arrays::NamedTuple)
    data = cat(values(arrays)...; dims=1)
    sizes = [size(a, 1) for a=values(arrays)]
    cmsm = cumsum(sizes)
    start = [1; cmsm[1:end-1] .+ 1]
    stop = cmsm
    ks = keys(arrays)
    #::NamedTuple{ks, NTuple{length(arrays), Tuple{UnitRange{Int64}, Tuple{Int64}}}}
    indicies = NamedTuple{ks}(
        (start[i]:stop[i], (sizes[i],)) for i=1:length(arrays)
    )
    ComponentTensor(data, indicies)
end
ComponentTensor(;arrays...) = ComponentTensor(NamedTuple(arrays))
data(ct::ComponentTensor) = getfield(ct, :data)
index(ct::ComponentTensor) = getfield(ct, :index)

Base.size(ct::ComponentTensor) = size(data(ct))
Base.length(ct::ComponentTensor) = length(data(ct))
Base.eltype(ct::ComponentTensor) = eltype(data(ct))
Base.ndims(ct::ComponentTensor) = ndims(data(ct))

function Base.getindex(ct::ComponentTensor, key::Symbol, inds...)
    ind1, shape = index(ct)[key]
    data(ct)[ind1, inds...]
    #reshape(view(ct.data, ind1, inds...), shape..., size(ct.A)[2:end]...)
end
function Base.getindex(ct::ComponentTensor, key::Symbol)
    if ndims(ct) > 1
        return ct[key, ntuple(_->:, ndims(ct)-1)...]
    else
        ind1, shape = index(ct)[key]
        return data(ct)[ind1]
    end
end
Base.getproperty(ct::ComponentTensor, key::Symbol) = view(ct, key, ntuple(_->:, ndims(ct)-1)...)
function Base.view(ct::ComponentTensor, ind1::Colon, inds...)
    ComponentTensor(view(data(ct), ind1, inds...), index(ct))
end
function Base.view(ct::ComponentTensor, key::Symbol, inds...)
    ind1, shape = index(ct)[key]
    view(data(ct), ind1, inds...)
    #reshape(view(ct.data, ind1, inds...), shape..., size(ct.A)[2:end]...)
end
Base.setindex!(ct::ComponentTensor, value, inds...) = data(ct)[inds...] = value
function Base.setindex!(ct::ComponentTensor, value, key::Symbol, inds...) 
    data(ct)[index(ct)[key][1], inds...] = value
end
function Base.setindex!(ct::ComponentTensor, value::ComponentTensor, inds...)
    @assert index(ct) == index(value)
    data(ct)[inds...] = data(value)
end
function Base.setindex!(ct::ComponentTensor, value::ComponentTensor, key::Symbol, inds...)
    data(ct)[index(ct)[key][1], inds...] = data(value)
end

function BatchComponentTensor(template::ComponentTensor, batch_dims...; array_fcn=zeros)
    ComponentTensor(array_fcn(size(template)..., batch_dims...), index(template))
end


function computeRange(ct::ComponentTensor, indkeys::Vector{Symbol})
    for i = 2:length(indkeys)
        @assert index(ct)[indkeys[i]][1].start == index(ct)[indkeys[i-1]][1].stop + 1
    end
    start = index(ct)[first(indkeys)][1].start
    stop = index(ct)[last(indkeys)][1].stop
    return start:stop
end

function Base.getindex(ct::ComponentTensor, keys::Vector{Symbol}, inds...)
    data(ct)[computeRange(ct, keys), inds...]
end
Base.getindex(ct::ComponentTensor, key::Vector{Symbol}) = ct[key, ntuple(_->:, ndims(ct)-1)...]

function Base.view(ct::ComponentTensor, indkeys::Vector{Symbol}, inds...)
    view(data(ct), computeRange(ct, indkeys), inds...)
end
function Base.view(ct::ComponentTensor, indkeys::Vector{Symbol})
    if ndims(ct) > 1
        return view(ct, indkeys, ntuple(_->:, ndims(ct)-1)...)
    else
        return view(data(ct), computeRange(ct, indkeys))
    end
end

function Base.view(ct::ComponentTensor, key::Symbol)
    if ndims(ct) > 1
        return view(ct, key, ntuple(_->:, ndims(ct)-1)...)
    else
        ind1, shape = index(ct)[key]
        return view(data(ct), ind1)
    end
end