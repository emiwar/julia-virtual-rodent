params = (
    network = (
        actor_size=[1024],
        critic_size=[1024, 1024],
        sigma_min=1f-2,
        sigma_max=2f0,#5f-1,
        latent_dimension=60
    ),
    physics = (
        n_physics_steps = 5,
        min_torso_z = 0.035,
        spawn_z_offset = 0.01,
        torque_control = true,
        body_scale = 1.0,
        timestep = 0.002,
        foot_mods = true,
        hip_mods = false
    ),
    reward = (
        alive_bonus = 0.1,
        control_cost = 0.001,
        forward_weight = 1.0
    ),
    training = (
        loss_weight_actor = 1.0,
        loss_weight_critic = 0.5,
        loss_weight_entropy = -0.2,
        n_miniepochs=2,
        learning_rate=1e-4,
        gamma=0.95,
        lambda=0.95,
        clip_range=0.2,
        checkpoint_interval=5000,
    ),
    rollout = (
        n_envs=512,
        n_steps_per_epoch=16,
        n_epochs=20_000,
        reset_on_epoch_start=false,
    )
)

import BSON
import CUDA
import Dates
import Statistics
import HDF5
import LinearAlgebra: norm
import MuJoCo
import PythonCall
import Wandb
using Flux
using ProgressMeter
using StaticArrays

include("../src/utils/profiler.jl")
include("../src/utils/load_dm_control_model.jl")
include("../src/utils/mujoco_quat.jl")
include("../src/utils/component_tensor.jl")
include("../src/utils/wandb_logger.jl")
include("../src/environments/mujoco_env.jl")
include("../src/environments/rodent_run_env.jl")
include("../src/collectors/batch_stepper.jl")
include("../src/collectors/mpi_stepper.jl")
include("../src/collectors/cuda_collector.jl")
include("../src/algorithms/ppo_loss.jl")
include("../src/networks/variational_enc_dec.jl")
include("../src/networks/state_invariant.jl")

template_env = RodentRunEnv(params)
ActorCritic = VariationalEncDec
pretrained_network, step = load_actor_critic_from_wandb("j0zwbgns")

stepper = BatchStepper(template_env, params.rollout.n_envs)
networks = StateInvariantActor(template_env, params)
collector = CuCollector(template_env, networks,
                        params.rollout.n_envs,
                        params.rollout.n_steps_per_epoch)
networks_gpu = networks |> Flux.gpu
opt_state = Flux.setup(Flux.Adam(params.training.learning_rate), networks_gpu)
run_name = "MaximizeXSpeed-$(Dates.now())" #ImitationWithAppendages
config = params_to_dict(params)

action_preprocess(action, state, params) = decoder_only(pretrained_network, state, action, params)
lg = Wandb.WandbLogger(project = "Rodent-Joystick",
                       name = run_name,
                       config = config)
mkdir("runs/checkpoints/$(run_name)")
@showprogress for epoch = 1:params.rollout.n_epochs
    
    lapTimer = LapTimer()
    collect_batch!(collector, stepper, params, lapTimer; action_preprocess) do state, params
        actor(networks_gpu, state, params)
    end
    ppo_log = ppo_update!(collector, networks_gpu, opt_state, params, lapTimer)
    lap(lapTimer, :logging_batch_stats)
    logdict = compute_batch_stats(collector)
    merge!(logdict, ppo_log)
    logdict["total_steps"] = epoch * params.rollout.n_envs * params.rollout.n_steps_per_epoch
    lap(lapTimer, :checkpointing)
    if epoch % params.training.checkpoint_interval == 0
        checkpoint_fn = "runs/checkpoints/$(run_name)/step-$(epoch).bson"
        BSON.bson(checkpoint_fn; actor_critic=Flux.cpu(actor_critic))
        lg.wrun.log_model(checkpoint_fn, "checkpoint-step-$(epoch).bson")
    end
    lap(lapTimer, :logging_submitting)
    merge!(logdict, to_stringdict(lapTimer))
    Wandb.log(lg, logdict)
end
Wandb.close(lg);
