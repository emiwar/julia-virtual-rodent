struct Rodent <: Walker
    model::MuJoCo.Model
    data::MuJoCo.Data
    sensorranges::Dict{String, UnitRange{Int64}}
    n_physics_steps::Int64
    min_torso_z::Float64
    spawn_z_offset::Float64
end

function Rodent(;body_scale::Float64, torque_control::Bool, foot_mods::Bool, hip_mods::Bool, 
                 timestep::Float64, min_torso_z::Float64, spawn_z_offset::Float64,
                 n_physics_steps::Int64)
    if body_scale != 1.0
        @warn "Code for rescaling imitation target removed from preprocessing, will use imitation target for scale=1.0." body_scale
    end
    model = dm_control_rodent(torque_actuators = torque_control,
                              foot_mods = foot_mods,
                              scale = body_scale,
                              hip_mods = hip_mods,
                              physics_timestep = timestep,
                              control_timestep = timestep*n_physics_steps)
    data = MuJoCo.init_data(model)
    sensors = SVector("velocimeter", "accelerometer", "gyro", "palm_L", "palm_R", "sole_L",
                      "sole_R", "torso")
    sensorranges = prepare_sensorranges(model, "walker/" .* sensors)
    Rodent(model, data, sensorranges, n_physics_steps, min_torso_z, spawn_z_offset)
end

function clone(rodent::Rodent)
    Rodent(rodent.model, MuJoCo.init_data(rodent.model), rodent.sensorranges,
           rodent.n_physics_steps, rodent.min_torso_z, rodent.spawn_z_offset)
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

function reset!(rodent::Rodent, start_qpos, start_qvel)
    MuJoCo.reset!(rodent.model, rodent.data)
    rodent.data.qpos .= start_qpos
    rodent.data.qpos[3] += rodent.spawn_z_offset
    rodent.data.qvel .= start_qvel
    #Run model forward to get correct initial state
    MuJoCo.forward!(rodent.model, rodent.data)
end

min_torso_z(rodent::Rodent) = rodent.min_torso_z

bodies_order(::Type{Rodent}) = SVector("torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
                                    "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
                                    "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
                                    "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R")

appendages_order(::Type{Rodent}) = SVector("lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull")

bodies_order(rodent::Rodent) = bodies_order(typeof(rodent))
appendages_order(rodent::Rodent) = appendages_order(typeof(rodent))

function appendage_pos(rodent::Rodent, appendage_name::String)
    body_xmat(rodent, "walker/torso")*(body_xpos(rodent, appendage_name) .- body_xpos(rodent, "walker/torso"))
end
@generated function appendages_pos(rodent::Rodent)
    :(hcat($((:(appendage_pos(rodent, "walker/" * $a)) for a=appendages_order(rodent))...)))
end

#Utils
torso_x(rodent::Rodent) = subtree_com(rodent, "walker/torso")[1]
torso_y(rodent::Rodent) = subtree_com(rodent, "walker/torso")[2]
torso_z(rodent::Rodent) = subtree_com(rodent, "walker/torso")[3]

function Base.show(io::IO, rodent::Rodent)
    compact = get(io, :compact, false)
    if compact
        print(io, "Rodent")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)Rodent")
        println(io, "$(indent)  n_physics_steps: ", rodent.n_physics_steps)
        println(io, "$(indent)  min_torso_z: ", rodent.min_torso_z)
        println(io, "$(indent)  spawn_z_offset: ", rodent.spawn_z_offset)
    end
end