params = (
    network = (
        encoder_size=[1024, 1024],
        decoder_size=[1024, 1024],
        critic_size=[1024, 1024],
        sigma_min=1f-2,
        sigma_max=5f-1,
        latent_dimension=60,
        bottleneck = :variational,
        decoder_type = :LSTM
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
        falloff = (
            com = 0.05,         #meter
            rotation = 0.5,     #radians
            joint = 2.0,        #root-sum-square of radians
            joint_vel = 5.0,    #root-sum-square of radians/sec
            appendages = 0.02,  #meter
            per_joint = 0.2,    #radians 
            per_joint_vel = 1.0 #radians / s
        ),
    ),
    imitation = (
        horizon = 5,
        max_target_distance = 1e-1,
        restart_on_reset = true
    ),
    training = (
        loss_weight_actor = 1.0,
        loss_weight_critic = 0.5,
        loss_weight_entropy = -0.2,#-0.00,#-0.5,
        loss_weight_kl = 0.1,
        n_miniepochs=1,
        learning_rate=1e-4,
        gamma=0.95,
        lambda=0.95,
        clip_range=0.2,
        checkpoint_interval=5000,
    ),
    rollout = (
        n_envs=512,
        n_steps_per_epoch=16,
        n_epochs=200_000,
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
import Random
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
include("../src/environments/imitation_trajectory.jl")
include("../src/environments/rodent_imitation_env.jl")
include("../src/collectors/batch_stepper.jl")
include("../src/collectors/mpi_stepper.jl")
include("../src/collectors/cuda_collector.jl")
include("../src/algorithms/ppo_loss.jl")
include("../src/networks/utils.jl")
include("../src/networks/variational_bottleneck.jl")
include("../src/networks/action_samplers.jl")
include("../src/networks/enc_dec.jl")

template_env = RodentImitationEnv(params)
stepper = BatchStepper(template_env, params.rollout.n_envs)

networks = EncDec(template_env, params)
networks_gpu = Flux.gpu(networks)

collector = CuCollector(template_env, networks,
                        params.rollout.n_envs,
                        params.rollout.n_steps_per_epoch)
networks_gpu = networks |> Flux.gpu
opt_state = Flux.setup(Flux.Adam(params.training.learning_rate), networks_gpu)


network_state_at_epoch_start = checkpoint_latent_state(networks_gpu)
lapTimer = LapTimer()
collect_batch!(collector, networks_gpu, stepper, params, lapTimer)
network_state_at_epoch_end = checkpoint_latent_state(networks_gpu)

ppo_log = ppo_update!(collector, networks_gpu, network_state_at_epoch_start, network_state_at_epoch_end,
                      opt_state, params, lapTimer)


network_state_at_epoch_start = checkpoint_latent_state(networks_gpu)
CUDA.seed!(1)
collect_batch!(collector, networks_gpu, stepper, params, lapTimer)
network_state_at_epoch_end = checkpoint_latent_state(networks_gpu)
restore_latent_state!(networks_gpu, network_state_at_epoch_start);
CUDA.seed!(1)
checkpoint_latent_state(networks_gpu)
@assert checkpoint_latent_state(networks_gpu) ≈ network_state_at_epoch_start
collect_batch!(collector, networks_gpu, stepper, params, lapTimer)
@assert checkpoint_latent_state(networks_gpu) ≈ network_state_at_epoch_end

dummy_inp = CUDA.randn(715)


state_before = checkpoint_latent_state(networks_gpu.encoder)
first_pass = networks_gpu.encoder(dummy_inp)
state_after = checkpoint_latent_state(networks_gpu.encoder)
restore_latent_state!(networks_gpu.encoder, state_before)
second_pass = networks_gpu.encoder(dummy_inp)
@assert first_pass ≈ second_pass
@assert state_after ≈ checkpoint_latent_state(networks_gpu.encoder)