params = (
    network = (
        encoder_size=[1024, 1024],
        decoder_size=[1024, 1024],
        critic_size=[1024, 1024],
        sigma_min=1f-2,
        sigma_max=5f-1,
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
        max_target_distance = 1e-1
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
        n_epochs=100,
        reset_on_epoch_start=false,
    )
)

include("../src/default_imports.jl")
include("../src/networks/variational_enc_dec_lstm.jl")
template_env = RodentImitationEnv(params)
stepper = BatchStepper(template_env, params.rollout.n_envs)

networks = VariationalEncDecLSTM(template_env, params)
collector = CuCollector(template_env, networks,
                        params.rollout.n_envs,
                        params.rollout.n_steps_per_epoch)
networks_gpu = networks |> Flux.gpu
reset_to_dim!(networks_gpu, params.rollout.n_envs)
opt_state = Flux.setup(Flux.Adam(params.training.learning_rate), networks_gpu)
run_name = "TryLSTM-$(Dates.now())" #ImitationWithAppendages
config = params_to_dict(params)

#lg = Wandb.WandbLogger(project = "Rodent-Imitation",
#                        name = run_name,
#                        config = config)
mkdir("runs/checkpoints/$(run_name)")
@showprogress for epoch = 1:params.rollout.n_epochs
    lapTimer = LapTimer()

    network_state_at_epoch_start = checkpoint_state(networks_gpu)
    collect_batch!(collector, stepper, params, lapTimer) do state, status, params
        actor(networks_gpu, state, status .== RUNNING, params)
    end
    network_state_at_epoch_end = checkpoint_state(networks_gpu)
    restore_state!(networks_gpu, network_state_at_epoch_start)
    actor(networks_gpu, collector.states, collector.status .== RUNNING, params)
    checkpoint_state(networks_gpu) == network_state_at_epoch_end
    
    before_miniepoch = ()->restore_state!(networks_gpu, network_state_at_epoch_start)
    ppo_log = ppo_update!(collector, networks_gpu, opt_state, params, lapTimer; before_miniepoch)
    if params.training.n_miniepochs == 1
        @assert checkpoint_state(networks_gpu) == network_state_at_epoch_end
    end
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



non_final_states = view(collector.states, :, :, 1:n_steps_per_batch)
actions_ = collector.actor_outputs.action |> array
latent_eps = collector.actor_outputs.latent_eps |> array
inv_reset_mask = view(collector.status .== RUNNING, :, 1:n_steps_per_batch)

A = checkpoint_state(networks_gpu)
actor(networks_gpu, non_final_states, inv_reset_mask, params, actions_, latent_eps)
B = checkpoint_state(networks_gpu)

restore_state!(networks_gpu, A)
actor(networks_gpu, non_final_states, inv_reset_mask, params, actions_, latent_eps)
C = checkpoint_state(networks_gpu)
@assert B == C

restore_state!(networks_gpu, A)
for t=1:n_steps_per_batch
    actor(networks_gpu, (@view collector.states[:, :, t]), (@view collector.status[:, t]), params, actions_[:, :, t], latent_eps[:, :, t])
end
D = checkpoint_state(networks_gpu)
@assert B == D