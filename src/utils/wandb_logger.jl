import Wandb
#import CUDA

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

function load_params_from_wandb(run_id, project="emiwar-team/Rodent-Imitation")
    api = Wandb.wandb.Api()
    ex_run = api.run("$project/$run_id")
    params_as_pydict = PythonCall.pyimport("json").loads(ex_run.json_config)
    function to_named_tuple(d, toplevel)
        if toplevel
            return NamedTuple(Symbol(k) => to_named_tuple(d[k]["value"], false) for k in d)
        elseif Bool(PythonCall.pytype(d) == PythonCall.pyimport("builtins").dict)
            return NamedTuple(Symbol(k) => to_named_tuple(d[k], false) for k in d)
        else
            return PythonCall.pyconvert(Any, d)
        end
    end
    params = to_named_tuple(params_as_pydict, true)
    return params
end

function load_actor_critic_from_wandb(run_id, checkpoint::Regex=r"\.bson"; project="emiwar-team/Rodent-Imitation")
    api = Wandb.wandb.Api()
    ex_run = api.run("$project/$run_id")
    all_artifacts = collect(ex_run.logged_artifacts())
    art_id = findlast(art->occursin(checkpoint, PythonCall.pyconvert(String, art.name)), all_artifacts)
    if art_id === nothing
        error("Could not find any checkpoint mathing '$checkpoint'.")
    end
    artifact = all_artifacts[art_id]
    artifact_name = PythonCall.pyconvert(String, artifact.name)
    println("Found mathing checkpoint: $artifact_name")
    regex_match = match(r"checkpoint-step-(\d+)", artifact_name)
    if isnothing(regex_match)
        error("Failed to extract epoch number from checkpoint filename $(artifact_name).")
    end
    epoch = parse(Int, regex_match.captures[1])

    artifact_folder = PythonCall.pyconvert(String, artifact.download())
    weights_file_name = artifact_folder * "/" * PythonCall.pyconvert(String, collect(artifact.files())[end].name)
    actor_critic = BSON.load(weights_file_name)[:actor_critic] |> Flux.gpu
    return actor_critic, epoch
end

function load_from_wandb(run_id, checkpoint::Regex=r"\.bson"; project="emiwar-team/Rodent-Imitation")
    api = Wandb.wandb.Api()
    ex_run = api.run("$project/$run_id")
    params_as_pydict = PythonCall.pyimport("json").loads(ex_run.json_config)
    function to_named_tuple(d, toplevel)
        if toplevel
            return NamedTuple(Symbol(k) => to_named_tuple(d[k]["value"], false) for k in d)
        elseif Bool(PythonCall.pytype(d) == PythonCall.pyimport("builtins").dict)
            return NamedTuple(Symbol(k) => to_named_tuple(d[k], false) for k in d)
        else
            return PythonCall.pyconvert(Any, d)
        end
    end
    params = to_named_tuple(params_as_pydict, true)
    all_artifacts = collect(ex_run.logged_artifacts())
    art_id = findlast(art->occursin(checkpoint, PythonCall.pyconvert(String, art.name)), all_artifacts)
    if art_id === nothing
        error("Could not find any checkpoint mathing '$checkpoint'.")
    end
    artifact = all_artifacts[art_id]
    println("Found mathing checkpoint: ", artifact.name)
    artifact_folder = PythonCall.pyconvert(String, artifact.download())
    weights_file_name = artifact_folder * "/" * PythonCall.pyconvert(String, collect(artifact.files())[end].name)
    return params, weights_file_name
end
