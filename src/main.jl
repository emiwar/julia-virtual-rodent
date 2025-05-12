import Dates
import TOML
include("utils/parse_config.jl")
include("utils/profiler.jl")
include("environments/environments.jl")
include("networks/networks.jl")
include("algorithms/algorithms.jl")

params = parse_config(ARGS)

#Setup the environment
walker = Environments.Rodent(;params.physics...)
reward_spec = Environments.EqualRewardWeights(;params.reward...)
target = Environments.load_imitation_target(walker)
template_env = Environments.ImitationEnv(walker, reward_spec, target; params.imitation...)
if haskey(params, :mod)
    if haskey(params.mod, :simplified_target) && params.mod.simplified_target
        template_env = Environments.SimplifiedTarget(template_env)
    end
    if haskey(params.mod, :imitation_speedup_range)
        template_env = Environments.FPSMod(template_env, params.mod.imitation_speedup_range)
    end
end

#MPI or local multithreading of environment
if params.rollout.use_mpi
    import MPI
    MPI.Init(threadlevel=:funneled)
    #This call is blocking on non-root processes
    env = Environments.MpiEnv(template_env, params.rollout.n_envs;
                              block_workers = true,
                              n_steps_per_epoch=params.rollout.n_steps_per_epoch)
else
    env = Environments.MultithreadEnv(template_env, params.rollout.n_envs)
end

#If MPI is used, these imports are only called on the root process  
import Wandb
import Flux
import BSON

#Setup the networks (encoder, decoder, critic)
networks = Networks.EncDec(template_env, params)
collector = Algorithms.CuCollector(env, networks, params.rollout.n_steps_per_epoch)
networks_gpu = networks |> Flux.gpu

#Setup wandb logging and checkpointing
include("utils/wandb_logger.jl")
const lg = Wandb.WandbLogger(project = params.wandb.project,
                             name = params.wandb.run_name,
                             config = params_to_dict(params))
mkdir("runs/checkpoints/$(params.wandb.run_name)")

#Callback called after each epoch
function logger(epoch, dict_to_log)
    Timers.lap(:checkpointing)
    #Checkpoint the network weights every `checkpoint_interval` epochs
    if epoch % params.training.checkpoint_interval == 0
        checkpoint_fn = "runs/checkpoints/$(params.wandb.run_name)/step-$(epoch).bson"
        networks_cpu = networks_gpu |> Flux.cpu
        BSON.bson(checkpoint_fn; actor_critic=networks_cpu,
                                 model_state=Flux.state(networks_cpu))
        lg.wrun.log_model(checkpoint_fn, "checkpoint-step-$(epoch).bson")
    end

    #Submit batch stats to WandB
    Timers.lap(:logging_submitting)
    merge!(dict_to_log, Timers.to_stringdict(Timers.default_timer))
    Wandb.log(lg, dict_to_log)
    Timers.reset!(Timers.default_timer)
end

#Run the training loop
Algorithms.ppo(collector, networks_gpu, params; logger)

#Cleanup
Wandb.close(lg)
if params.rollout.use_mpi
    MPI.Abort(MPI.COMM_WORLD, 0)
    MPI.Finalize()
end