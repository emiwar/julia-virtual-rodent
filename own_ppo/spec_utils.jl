function prealloc_spec(spec::Rectangle, batch_dims, eltype::Type; array_fcn=zeros)
    return array_fcn(eltype, length(spec.a), batch_dims...)
end
function prealloc_spec(spec::Rectangle, batch_dims; array_fcn=zeros)
    eltype = spec |> Base.eltype |> Base.eltype
    return prealloc_spec(spec, batch_dims, eltype; array_fcn)
end

function prealloc_spec(spec::ClosedInterval, batch_dims; array_fcn=zeros)
    eltype = spec |> Base.eltype
    return prealloc_spec(spec, batch_dims, eltype; array_fcn)
end
function prealloc_spec(spec::ClosedInterval, batch_dims, eltype::Type; array_fcn=zeros)
    return array_fcn(eltype, batch_dims...)
end

function prealloc_spec(spec::NamedTuple, batch_dims; array_fcn=zeros)
    map(s->prealloc_spec(s, batch_dims; array_fcn), spec)
end

function prealloc_spec(spec::NamedTuple, batch_dims, eltype::Type; array_fcn=zeros)
    map(s->prealloc_spec(s, batch_dims, eltype; array_fcn), spec)
end
