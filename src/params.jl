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
        min_torso_z = 0.03,
        spawn_z_offset = 0.01,
        torque_control = false,
        body_scale = 1.0,
        timestep = 0.002,
        foot_mods = true,
        hip_mods = true
    ),
    reward = (
        alive_bonus = 0.1,
        control_cost = 0.001,
        falloff = (
            com = 0.05,       #meter
            rotation = 0.5,   #radians
            joint = 2.0,      #root-sum-square of radians
            appendages = 0.02 #meter
        ),
    ),
    imitation = (
        horizon = 5,
        max_target_distance = 2e-1
    ),
    training = (
        loss_weight_actor = 1.0,
        loss_weight_critic = 0.5,
        loss_weight_entropy = -0.2,#-0.00,#-0.5,
        n_miniepochs=2,
        learning_rate=1e-4,
        gamma=0.95,
        lambda=0.95,
        clip_range=0.2,
        checkpoint_interval=5000,
    ),
    rollout = (
        n_envs=4096,
        n_steps_per_epoch=16,
        n_epochs=100_000,
        reset_on_epoch_start=false,
    )
)
