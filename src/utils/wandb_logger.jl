import Wandb
import CUDA

function log_to_wandb(lg::Wandb.WandbLogger, key, val; quantiles=0:0.25:1)
    if ndims(val) >= 1
        inds = Int.(ceil.(CUDA.cu(quantiles) .* (length(val) .- 1))) .+ 1
        qv = CUDA.sort(view(val, :))[inds] |> Array
        Wandb.log(lg, Dict("$(key)_q$(100q)" => v for (q,v) in zip(quantiles, qv)), commit=false)
    else
        Wandb.log(lg, Dict(key=>val), commit=false)
    end
end
