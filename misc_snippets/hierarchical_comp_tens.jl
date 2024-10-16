s = state(env, params)

function ct(nt::NamedTuple)
    counter = Ref(1)
    function computeRanges(a::AbstractArray)
        old_ind = counter[]
        counter[] += length(a)
        return old_ind:(counter[]-1)
    end
    computeRanges(nt::NamedTuple) = map(computeRanges, nt)
    ranges = computeRanges(nt)
    data = zeros(counter[])
    function copyData(range::NamedTuple, arrays::NamedTuple)
        for (key, value) in pairs(arrays)
            if value isa AbstractArray
                data[range[key]] .= reshape(value, :)
            else
                copyData(range[key], arrays[key])
            end
        end
    end
    copyData(ranges, nt)
    return ranges, data
end

ct(s)