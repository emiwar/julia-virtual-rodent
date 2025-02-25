include("../src/default_imports_mpi.jl")
params = (
    network = (
        encoder_size=[1024, 1024],
        decoder_size=[1024, 1024],
        critic_size=[1024, 1024],
        sigma_min=1f-2,
        sigma_max=5f-1,
        latent_dimension=60,
        bottleneck=:variational,
        decoder_type=:LSTM,
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
        control_cost = 0.015,
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
        n_envs=4096,
        n_steps_per_epoch=16,
        n_epochs=500000,
        reset_on_epoch_start=false,
    ),
    wandb = (
        project = "Rodent-Imitation",
        run_name = "BigRefactor-$(Dates.now())",
    )
)

env = RodentImitationEnv(params)
ppo(env, params; use_mpi=isdefined(Main, :MPI))
