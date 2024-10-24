import Wandb
import CUDA

function log_to_wandb(lg::Wandb.WandbLogger, key, val; quantiles=0:0.25:1)
    if ndims(val) >= 1
        Wandb.log(lg, quantile_dict(key, val; quantiles), commit=false)
    else
        Wandb.log(lg, Dict(key=>val), commit=false)
    end
end

function quantile_dict(prefix, val; quantiles=0:0.25:1)
    if length(val) == 0
        return Dict{String, eltype(val)}()
    end
    inds = Int.(ceil.(CUDA.cu(quantiles) .* (length(val) .- 1))) .+ 1
    qv = CUDA.sort(view(val, :))[inds] |> Array
    Dict("$(prefix)_q$(100q)" => v for (q,v) in zip(quantiles, qv))
end

function params_to_dict(params)
    to_dict(v) = v
    to_dict(p::NamedTuple) = Dict(string(k)=>to_dict(v) for (k,v) in pairs(p))
    to_dict(params)
end