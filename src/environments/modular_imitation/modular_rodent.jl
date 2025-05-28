struct ModularRodent{S} <: Walker
    model::MuJoCo.Model
    data::MuJoCo.Data
    sensors::S
    n_physics_steps::Int64
    min_torso_z::Float64
    spawn_z_offset::Float64
end

function ModularRodent(;min_torso_z::Float64, spawn_z_offset::Float64, n_physics_steps::Int64)
    model_path = joinpath(@__DIR__, "../assets/rodent_extra_sensors.xml")
    model = MuJoCo.load_model(model_path)
    data = MuJoCo.init_data(model)
    sensors = ComponentArray(view(data.sensordata, :), create_sensorindex(model))
    ModularRodent(model, data, sensors, n_physics_steps, min_torso_z, spawn_z_offset)
end

function clone(rodent::ModularRodent)
    new_data = MuJoCo.init_data(rodent.model)
    ModularRodent(rodent.model, new_data,
                  ComponentArray(view(new_data.sensordata, :), create_sensorindex(rodent.model)),
                  rodent.n_physics_steps, rodent.min_torso_z, rodent.spawn_z_offset)
end

function proprioception(rodent::ModularRodent)
    (
        hand_L = (
            palm_contact = sensor(rodent, "hand_L/palm_contact" |> Symbol |> Val),
            egocentric_hand_pos = 20.0 * SVector{3}(sensor(rodent, "hand_L/egocentric_hand_pos" |> Symbol |> Val)),
            elbow_angle = sensor(rodent, "hand_L/elbow_angle" |> Symbol |> Val),
            wrist_angle = sensor(rodent, "hand_L/wrist_angle" |> Symbol |> Val),
            finger_angle = sensor(rodent, "hand_L/finger_angle" |> Symbol |> Val),
            finger_limit = sensor(rodent, "hand_L/finger_limit" |> Symbol |> Val),
            wrist_limit = sensor(rodent, "hand_L/wrist_limit" |> Symbol |> Val),
            xaxis = sensor(rodent, "hand_L/xaxis" |> Symbol |> Val),
            finger_force = sensor(rodent, "hand_L/finger_force" |> Symbol |> Val),
            wrist_force = sensor(rodent, "hand_L/wrist_force" |> Symbol |> Val),
        ),
        arm_L = (
            egocentric_hand_pos = 20.0 * SVector{3}(sensor(rodent, "arm_L/egocentric_hand_pos" |> Symbol |> Val)),
            egocentric_elbow_pos = 20.0 * SVector{3}(sensor(rodent, "arm_L/egocentric_elbow_pos" |> Symbol |> Val)),
            wrist_angle = sensor(rodent, "arm_L/wrist_angle" |> Symbol |> Val),
            elbow_angle = sensor(rodent, "arm_L/elbow_angle" |> Symbol |> Val),
            shoulder_sup = sensor(rodent, "arm_L/shoulder_sup" |> Symbol |> Val),
            shoulder = sensor(rodent, "arm_L/shoulder" |> Symbol |> Val),
            scapula_extend = sensor(rodent, "arm_L/scapula_extend" |> Symbol |> Val),
            scapula_abduct = sensor(rodent, "arm_L/scapula_abduct" |> Symbol |> Val),
            scapula_supinate = sensor(rodent, "arm_L/scapula_supinate" |> Symbol |> Val),
            elbow_limit = sensor(rodent, "arm_L/elbow_limit" |> Symbol |> Val),
            wrist_force = sensor(rodent, "arm_L/wrist_force" |> Symbol |> Val),
            elbow_force = sensor(rodent, "arm_L/elbow_force" |> Symbol |> Val),
            shoulder_force = sensor(rodent, "arm_L/shoulder_force" |> Symbol |> Val),
        ),
        hand_R = (
            palm_contact = sensor(rodent, "hand_R/palm_contact" |> Symbol |> Val),
            egocentric_hand_pos = 20.0 * SVector{3}(sensor(rodent, "hand_R/egocentric_hand_pos" |> Symbol |> Val)),
            elbow_angle = sensor(rodent, "hand_R/elbow_angle" |> Symbol |> Val),
            wrist_angle = sensor(rodent, "hand_R/wrist_angle" |> Symbol |> Val),
            finger_angle = sensor(rodent, "hand_R/finger_angle" |> Symbol |> Val),
            finger_limit = sensor(rodent, "hand_R/finger_limit" |> Symbol |> Val),
            wrist_limit = sensor(rodent, "hand_R/wrist_limit" |> Symbol |> Val),
            xaxis = sensor(rodent, "hand_R/xaxis" |> Symbol |> Val),
            finger_force = sensor(rodent, "hand_R/finger_force" |> Symbol |> Val),
            wrist_force = sensor(rodent, "hand_R/wrist_force" |> Symbol |> Val),
        ),
        arm_R = (
            egocentric_hand_pos = 20.0 * SVector{3}(sensor(rodent, "arm_R/egocentric_hand_pos" |> Symbol |> Val)),
            egocentric_elbow_pos = 20.0 * SVector{3}(sensor(rodent, "arm_R/egocentric_elbow_pos" |> Symbol |> Val)),
            wrist_angle = sensor(rodent, "arm_R/wrist_angle" |> Symbol |> Val),
            elbow_angle = sensor(rodent, "arm_R/elbow_angle" |> Symbol |> Val),
            shoulder_sup = sensor(rodent, "arm_R/shoulder_sup" |> Symbol |> Val),
            shoulder = sensor(rodent, "arm_R/shoulder" |> Symbol |> Val),
            scapula_extend = sensor(rodent, "arm_R/scapula_extend" |> Symbol |> Val),
            scapula_abduct = sensor(rodent, "arm_R/scapula_abduct" |> Symbol |> Val),
            scapula_supinate = sensor(rodent, "arm_R/scapula_supinate" |> Symbol |> Val),
            elbow_limit = sensor(rodent, "arm_R/elbow_limit" |> Symbol |> Val),
            wrist_force = sensor(rodent, "arm_R/wrist_force" |> Symbol |> Val),
            elbow_force = sensor(rodent, "arm_R/elbow_force" |> Symbol |> Val),
            shoulder_force = sensor(rodent, "arm_R/shoulder_force" |> Symbol |> Val),
        ),
        foot_L = (
            sole_contact = sensor(rodent, "foot_L/sole_contact" |> Symbol |> Val),
            heel_contact = sensor(rodent, "foot_L/heel_contact" |> Symbol |> Val),
            egocentric_foot_pos = 20.0 * SVector{3}(sensor(rodent, "foot_L/egocentric_foot_pos" |> Symbol |> Val)),
            knee_angle = sensor(rodent, "foot_L/knee_angle" |> Symbol |> Val),
            ankle_angle = sensor(rodent, "foot_L/ankle_angle" |> Symbol |> Val),
            toe_angle = sensor(rodent, "foot_L/toe_angle" |> Symbol |> Val),
            toe_limit = sensor(rodent, "foot_L/toe_limit" |> Symbol |> Val),
            foot_limit = sensor(rodent, "foot_L/foot_limit" |> Symbol |> Val),
            xaxis = sensor(rodent, "foot_L/xaxis" |> Symbol |> Val),
            toe_force = sensor(rodent, "foot_L/toe_force" |> Symbol |> Val),
            ankle_force = sensor(rodent, "foot_L/ankle_force" |> Symbol |> Val),
        ),
        leg_L = (
            egocentric_foot_pos = 20.0 * SVector{3}(sensor(rodent, "leg_L/egocentric_foot_pos" |> Symbol |> Val)),
            hip_supinate = sensor(rodent, "leg_L/hip_supinate" |> Symbol |> Val),
            hip_abduct = sensor(rodent, "leg_L/hip_abduct" |> Symbol |> Val),
            hip_extend = sensor(rodent, "leg_L/hip_extend" |> Symbol |> Val),
            knee_angle = sensor(rodent, "leg_L/knee_angle" |> Symbol |> Val),
            ankle_angle = sensor(rodent, "leg_L/ankle_angle" |> Symbol |> Val),
            knee_limit = sensor(rodent, "leg_L/knee_limit" |> Symbol |> Val),
            ankle_force = sensor(rodent, "leg_L/ankle_force" |> Symbol |> Val),
            knee_force = sensor(rodent, "leg_L/knee_force" |> Symbol |> Val),
            hip_force = sensor(rodent, "leg_L/hip_force" |> Symbol |> Val),
        ),
        foot_R = (
            sole_contact = sensor(rodent, "foot_R/sole_contact" |> Symbol |> Val),
            heel_contact = sensor(rodent, "foot_R/heel_contact" |> Symbol |> Val),
            egocentric_foot_pos = 20.0 * SVector{3}(sensor(rodent, "foot_R/egocentric_foot_pos" |> Symbol |> Val)),
            knee_angle = sensor(rodent, "foot_R/knee_angle" |> Symbol |> Val),
            ankle_angle = sensor(rodent, "foot_R/ankle_angle" |> Symbol |> Val),
            toe_angle = sensor(rodent, "foot_R/toe_angle" |> Symbol |> Val),
            toe_limit = sensor(rodent, "foot_R/toe_limit" |> Symbol |> Val),
            foot_limit = sensor(rodent, "foot_R/foot_limit" |> Symbol |> Val),
            xaxis = sensor(rodent, "foot_R/xaxis" |> Symbol |> Val),
            toe_force = sensor(rodent, "foot_R/toe_force" |> Symbol |> Val),
            ankle_force = sensor(rodent, "foot_R/ankle_force" |> Symbol |> Val),
        ),
        leg_R = (
            egocentric_foot_pos = 20.0 * SVector{3}(sensor(rodent, "leg_R/egocentric_foot_pos" |> Symbol |> Val)),
            hip_supinate = sensor(rodent, "leg_R/hip_supinate" |> Symbol |> Val),
            hip_abduct = sensor(rodent, "leg_R/hip_abduct" |> Symbol |> Val),
            hip_extend = sensor(rodent, "leg_R/hip_extend" |> Symbol |> Val),
            knee_angle = sensor(rodent, "leg_R/knee_angle" |> Symbol |> Val),
            ankle_angle = sensor(rodent, "leg_R/ankle_angle" |> Symbol |> Val),
            knee_limit = sensor(rodent, "leg_R/knee_limit" |> Symbol |> Val),
            ankle_force = sensor(rodent, "leg_R/ankle_force" |> Symbol |> Val),
            knee_force = sensor(rodent, "leg_R/knee_force" |> Symbol |> Val),
            hip_force = sensor(rodent, "leg_R/hip_force" |> Symbol |> Val),
        ),
        torso = (
            accelerometer = 0.1 * SVector{3}(sensor(rodent, "torso/accelerometer" |> Symbol |> Val)),
            lumbar_extend = sensor(rodent, "torso/lumbar_extend" |> Symbol |> Val),
            lumbar_bend = sensor(rodent, "torso/lumbar_bend" |> Symbol |> Val),
            lumbar_twist = sensor(rodent, "torso/lumbar_twist" |> Symbol |> Val),
            linvel = sensor(rodent, "torso/linvel" |> Symbol |> Val),
        ),
        head = (
            accelerometer = 0.1 * SVector{3}(sensor(rodent, "head/accelerometer" |> Symbol |> Val)),
            egocentric_pos = 20.0 * SVector{3}(sensor(rodent, "head/egocentric_pos" |> Symbol |> Val)),
            xaxis = sensor(rodent, "head/xaxis" |> Symbol |> Val),
            zaxis = sensor(rodent, "head/zaxis" |> Symbol |> Val),
            neck_force = 0.01 * SVector{3}(sensor(rodent, "head/neck_force" |> Symbol |> Val)),
            atlas = sensor(rodent, "head/atlas" |> Symbol |> Val),
            atlant_extend = sensor(rodent, "head/atlant_extend" |> Symbol |> Val),
            mandible = sensor(rodent, "head/mandible" |> Symbol |> Val),
            cervical_extend = sensor(rodent, "head/cervical_extend" |> Symbol |> Val),
            cervical_bend = sensor(rodent, "head/cervical_bend" |> Symbol |> Val),
            cervical_twist = sensor(rodent, "head/cervical_twist" |> Symbol |> Val),
        )
    )
end

function reset!(rodent::ModularRodent, start_qpos, start_qvel)
    MuJoCo.reset!(rodent.model, rodent.data)
    rodent.data.qpos .= start_qpos
    rodent.data.qpos[3] += rodent.spawn_z_offset
    rodent.data.qvel .= start_qvel
    #Run model forward to get correct initial state
    MuJoCo.forward!(rodent.model, rodent.data)
end

min_torso_z(rodent::ModularRodent) = rodent.min_torso_z

bodies_order(::Type{R}) where R<:ModularRodent = SVector("torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
                                    "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
                                    "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
                                    "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R")

appendages_order(::Type{R}) where R<:ModularRodent = SVector("lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull")
bodies_order(rodent::ModularRodent) = bodies_order(typeof(rodent))
appendages_order(rodent::ModularRodent) = appendages_order(typeof(rodent))

function Base.show(io::IO, rodent::ModularRodent)
    compact = get(io, :compact, false)
    if compact
        print(io, "ModularRodent")
    else
        indent = " " ^ get(io, :indent, 0)
        println(io, "$(indent)ModularRodent")
        println(io, "$(indent)  n_physics_steps: ", rodent.n_physics_steps)
        println(io, "$(indent)  min_torso_z: ", rodent.min_torso_z)
        println(io, "$(indent)  spawn_z_offset: ", rodent.spawn_z_offset)
    end
end