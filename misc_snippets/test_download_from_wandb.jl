import BSON
import Flux
import CUDA
import MuJoCo
import Wandb
import PythonCall: pytype, pyconvert, pyimport
include("../src/utils/component_tensor.jl")
include("../src/environments/rodent_imitation_env.jl")
include("../src/algorithms/ppo_networks.jl")
MuJoCo.init_visualiser()

project="emiwar-team/Rodent-Imitation"
run_id = "vo055vvi"
checkpoint_regex = r"checkpoint-.*" #r"step-50000"

api = Wandb.wandb.Api()
ex_run = api.run("$project/$run_id")
params_as_pydict = PythonCall.pyimport("json").loads(ex_run.json_config)
function to_named_tuple(d, toplevel)
    if toplevel
        return NamedTuple(Symbol(k) => to_named_tuple(d[k]["value"], false) for k in d)
    elseif Bool(pytype(d) == pyimport("builtins").dict)
        return NamedTuple(Symbol(k) => to_named_tuple(d[k], false) for k in d)
    else
        return pyconvert(Any, d)
    end
end
params = to_named_tuple(params_as_pydict, true)
all_artifacts = collect(ex_run.logged_artifacts())
art_id = findlast(art->occursin(checkpoint_regex, pyconvert(String, art.name)), all_artifacts)
artifact = all_artifacts[art_id]
artifact.name
artifact_folder = pyconvert(String, artifact.download())
weights_file_name = artifact_folder * "/" * pyconvert(String, collect(artifact.files())[end].name)

