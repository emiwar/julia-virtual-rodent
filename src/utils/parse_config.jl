function _set_rec!(d::Dict, k::Vector, v)
    first_k = first(k)
    if !haskey(d, first_k)
        error("Unknown parameter $first_k")
    end
    if length(k) == 1
        d[first_k] = d[first_k] isa String ? v : parse(typeof(d[first_k]), v)
    else
        _set_rec!(d[first_k], k[2:end], v)
    end
end

function dict_to_named_tuple(params::Dict)
    to_nt(v) = v
    to_nt(p::Dict) = NamedTuple(Symbol(k)=>to_nt(v) for (k,v) in pairs(p))
    to_nt(params)
end

function parse_config(args)
    params = Dict{String, Any}()
    override_key = ""
    for arg in args
        if startswith(arg, "-")
            override_key == "" || error("Expected value for $override_key, not $arg.")
            override_key = arg[2:end]
        elseif override_key == ""
            d = TOML.parsefile(arg)
            merge!(params, d)
        else
            _set_rec!(params, split(override_key, "."), arg)
            override_key = ""
        end
    end
    params["wandb"]["run_name"] = replace(params["wandb"]["run_name"], "{NOW}"=>string(Dates.now()))
    to_sym = [("network", "bottleneck"), ("network", "decoder_type"), ("network", "actor_type")]
    for (l1, l2) in to_sym
        if haskey(params, l1) && haskey(params[l1], l2)
            params[l1][l2] = Symbol(params[l1][l2])
        end
    end
    return dict_to_named_tuple(params)
end
