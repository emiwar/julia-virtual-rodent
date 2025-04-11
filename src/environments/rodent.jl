abstract type Walker end

struct Rodent <: Walker
    model::MuJoCo.Model
    data::MuJoCo.Data
    sensorranges::Dict{String, UnitRange{Int64}}
end

function Rodent(params)
    if params.physics.body_scale != 1.0
        @warn "Code for rescaling imitation target removed from preprocessing, will use imitation target for scale=1.0." params.physics.body_scale
    end
    model = dm_control_rodent(torque_actuators = params.physics.torque_control,
                              foot_mods = params.physics.foot_mods,
                              scale = params.physics.body_scale,
                              hip_mods = params.physics.hip_mods,
                              physics_timestep = params.physics.timestep,
                              control_timestep = params.physics.timestep * params.physics.n_physics_steps)
    data = MuJoCo.init_data(model)
    sensorranges = prepare_sensorranges(model, "walker/" .* ["accelerometer", "velocimeter",
                                                             "gyro", "palm_L", "palm_R",
                                                             "sole_L", "sole_R", "torso"])
    Rodent(model, data, sensorranges)
end

function clone(rodent::Rodent)
    Rodent(rodent.model, MuJoCo.init_data(env.model), rodent.sensorranges)
end

function proprioception(rodent::Rodent)
    (
        joints = (@view rodent.data.qpos[8:end]),
        joint_vel = (@view rodent.data.qvel[7:end]),
        actuations = (@view rodent.data.act[:]),
        head = (
            velocity = sensor(rodent, "walker/velocimeter"),
            accel = sensor(rodent, "walker/accelerometer"),
            gyro = sensor(rodent, "walker/gyro")
        ),
        torso = (
            velocity = sensor(rodent, "walker/torso"),
            xmat = reshape(body_xmat(rodent, "walker/torso"), :),
            com = subtree_com(rodent, "walker/torso")
        ),
        paw_contacts = (
            palm_L = sensor(rodent, "walker/palm_L"),
            palm_R = sensor(rodent, "walker/palm_R"),
            sole_L = sensor(rodent, "walker/sole_L"),
            sole_R = sensor(rodent, "walker/sole_R")
        )
    )
end

#Appendages
bodies_order() = SVector("torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
                  "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
                  "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
                  "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R")

appendages_order() = SVector("lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull") 

function appendage_pos(rodent::Rodent, appendage_name::String)
    body_xmat(rodent, "walker/torso")*(body_xpos(rodent, appendage_name) .- body_xpos(rodent, "walker/torso"))
end
@generated function appendages_pos(rodent::Rodent)
    :(hcat($((:(appendage_pos(rodent, "walker/" * $a)) for a=appendages_order())...)))
end

#Utils
torso_x(rodent::Rodent) = subtree_com(rodent, "walker/torso")[1]
torso_y(rodent::Rodent) = subtree_com(rodent, "walker/torso")[2]
torso_z(rodent::Rodent) = subtree_com(rodent, "walker/torso")[3]

