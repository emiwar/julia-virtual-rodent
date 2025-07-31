@concrete struct ModularRodent <: Walker
    model::MuJoCo.Model
    data::MuJoCo.Data
    sensors<:ComponentVector
    actuators<:ComponentVector
    n_physics_steps::Int64
    min_torso_z::Float64
    spawn_z_offset::Float64
end

#TODO: Create a destrutor that deallocates the UnsafeArrays `sensors` and `actuators` safely,
#before deallocating the `data` object

function ModularRodent(;fixed_root::Bool=false, min_torso_z::Float64, spawn_z_offset::Float64, n_physics_steps::Int64)
    if fixed_root
        model_path = joinpath(@__DIR__, "..", "assets", "rodent_extra_sensors_fixed_root.xml")
    else
        model_path = joinpath(@__DIR__, "..", "assets", "rodent_extra_sensors.xml")
    end
    model = MuJoCo.load_model(model_path)
    data = MuJoCo.init_data(model)
    sensors = ComponentArray(view(data.sensordata, :), create_sensorindex(model))
    actuators = ComponentArray(view(data.ctrl, :), create_actuatorindex(model))
    ModularRodent(model, data, sensors, actuators, n_physics_steps, min_torso_z, spawn_z_offset)
end

function clone(rodent::ModularRodent)
    new_data = MuJoCo.init_data(rodent.model)
    ModularRodent(rodent.model, new_data,
                  ComponentArray(view(new_data.sensordata, :), create_sensorindex(rodent.model)),
                  ComponentArray(view(new_data.ctrl, :), create_actuatorindex(rodent.model)),
                  rodent.n_physics_steps, rodent.min_torso_z, rodent.spawn_z_offset)
end

function proprioception(rodent::ModularRodent)
    (
        hand_L = (
            palm_contact = sensor(rodent, "hand_L/palm_contact" |> Symbol |> Val),
            egocentric_hand_pos = body_relpos(rodent, "walker/hand_L") .* 20,
            wrist_angle = sensor(rodent, "hand_L/wrist_angle" |> Symbol |> Val),
            finger_angle = sensor(rodent, "hand_L/finger_angle" |> Symbol |> Val),
            finger_limit = sensor(rodent, "hand_L/finger_limit" |> Symbol |> Val),
            wrist_limit = sensor(rodent, "hand_L/wrist_limit" |> Symbol |> Val),
            xaxis = sensor(rodent, "hand_L/xaxis" |> Symbol |> Val),
            finger_force = sensor(rodent, "hand_L/finger_force" |> Symbol |> Val),
            wrist_force = sensor(rodent, "hand_L/wrist_force" |> Symbol |> Val),
        ),
        arm_L = (
            egocentric_elbow_pos = 20.0 * SVector{3}(sensor(rodent, "arm_L/egocentric_elbow_pos" |> Symbol |> Val)),
            elbow_angle = sensor(rodent, "arm_L/elbow_angle" |> Symbol |> Val),
            shoulder_sup = sensor(rodent, "arm_L/shoulder_sup" |> Symbol |> Val),
            shoulder = sensor(rodent, "arm_L/shoulder" |> Symbol |> Val),
            scapula_extend = sensor(rodent, "arm_L/scapula_extend" |> Symbol |> Val),
            scapula_abduct = sensor(rodent, "arm_L/scapula_abduct" |> Symbol |> Val),
            scapula_supinate = sensor(rodent, "arm_L/scapula_supinate" |> Symbol |> Val),
            elbow_limit = sensor(rodent, "arm_L/elbow_limit" |> Symbol |> Val),
            elbow_force = sensor(rodent, "arm_L/elbow_force" |> Symbol |> Val),
            shoulder_force = sensor(rodent, "arm_L/shoulder_force" |> Symbol |> Val),
            elbow_height = Environments.site_z(rodent, "walker/elbow_L") * 20.0,
            shoulder_height = Environments.site_z(rodent, "walker/shoulder_L") * 20.0,
            hand_linvel = torso_yawmat(rodent) * SVector{3}(sensor(rodent, "arm_L/hand_linvel" |> Symbol |> Val)),
        ),
        hand_R = (
            palm_contact = sensor(rodent, "hand_R/palm_contact" |> Symbol |> Val),
            egocentric_hand_pos = body_relpos(rodent, "walker/hand_R") .* 20,
            wrist_angle = sensor(rodent, "hand_R/wrist_angle" |> Symbol |> Val),
            finger_angle = sensor(rodent, "hand_R/finger_angle" |> Symbol |> Val),
            finger_limit = sensor(rodent, "hand_R/finger_limit" |> Symbol |> Val),
            wrist_limit = sensor(rodent, "hand_R/wrist_limit" |> Symbol |> Val),
            xaxis = sensor(rodent, "hand_R/xaxis" |> Symbol |> Val),
            finger_force = sensor(rodent, "hand_R/finger_force" |> Symbol |> Val),
            wrist_force = sensor(rodent, "hand_R/wrist_force" |> Symbol |> Val),
        ),
        arm_R = (
            egocentric_elbow_pos = 20.0 * SVector{3}(sensor(rodent, "arm_R/egocentric_elbow_pos" |> Symbol |> Val)),
            elbow_angle = sensor(rodent, "arm_R/elbow_angle" |> Symbol |> Val),
            shoulder_sup = sensor(rodent, "arm_R/shoulder_sup" |> Symbol |> Val),
            shoulder = sensor(rodent, "arm_R/shoulder" |> Symbol |> Val),
            scapula_extend = sensor(rodent, "arm_R/scapula_extend" |> Symbol |> Val),
            scapula_abduct = sensor(rodent, "arm_R/scapula_abduct" |> Symbol |> Val),
            scapula_supinate = sensor(rodent, "arm_R/scapula_supinate" |> Symbol |> Val),
            elbow_limit = sensor(rodent, "arm_R/elbow_limit" |> Symbol |> Val),
            elbow_force = sensor(rodent, "arm_R/elbow_force" |> Symbol |> Val),
            shoulder_force = sensor(rodent, "arm_R/shoulder_force" |> Symbol |> Val),
            elbow_height = Environments.site_z(rodent, "walker/elbow_R") * 20.0,
            shoulder_height = Environments.site_z(rodent, "walker/shoulder_R") * 20.0,
            hand_linvel = torso_yawmat(rodent) * SVector{3}(sensor(rodent, "arm_R/hand_linvel" |> Symbol |> Val)),
        ),
        foot_L = (
            sole_contact = sensor(rodent, "foot_L/sole_contact" |> Symbol |> Val),
            heel_contact = sensor(rodent, "foot_L/heel_contact" |> Symbol |> Val),
            egocentric_foot_pos = body_relpos(rodent, "walker/foot_L") .* 20,
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
            egocentric_knee_pos = 20.0 * SVector{3}(sensor(rodent, "leg_L/egocentric_knee_pos" |> Symbol |> Val)),
            hip_supinate = sensor(rodent, "leg_L/hip_supinate" |> Symbol |> Val),
            hip_abduct = sensor(rodent, "leg_L/hip_abduct" |> Symbol |> Val),
            hip_extend = sensor(rodent, "leg_L/hip_extend" |> Symbol |> Val),
            knee_angle = sensor(rodent, "leg_L/knee_angle" |> Symbol |> Val),
            knee_limit = sensor(rodent, "leg_L/knee_limit" |> Symbol |> Val),
            knee_force = sensor(rodent, "leg_L/knee_force" |> Symbol |> Val),
            hip_force = sensor(rodent, "leg_L/hip_force" |> Symbol |> Val),
            pelvis_zaxis = torso_yawmat(rodent) * SVector{3}(sensor(rodent, "pelvis/zaxis" |> Symbol |> Val)),
            hip_height = site_z(rodent, "walker/hip_L") .* 20.0,
            hip_accelerometer = 0.1 * SVector{3}(sensor(rodent, "leg_L/hip_accelerometer" |> Symbol |> Val)),
            hip_gyro = 0.1 * SVector{3}(sensor(rodent, "leg_L/hip_gyro" |> Symbol |> Val)),
        ),
        foot_R = (
            sole_contact = sensor(rodent, "foot_R/sole_contact" |> Symbol |> Val),
            heel_contact = sensor(rodent, "foot_R/heel_contact" |> Symbol |> Val),
            egocentric_foot_pos = body_relpos(rodent, "walker/foot_R") .* 20,
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
            egocentric_knee_pos = 20.0 * SVector{3}(sensor(rodent, "leg_R/egocentric_knee_pos" |> Symbol |> Val)),
            hip_supinate = sensor(rodent, "leg_R/hip_supinate" |> Symbol |> Val),
            hip_abduct = sensor(rodent, "leg_R/hip_abduct" |> Symbol |> Val),
            hip_extend = sensor(rodent, "leg_R/hip_extend" |> Symbol |> Val),
            knee_angle = sensor(rodent, "leg_R/knee_angle" |> Symbol |> Val),
            ankle_angle = sensor(rodent, "leg_R/ankle_angle" |> Symbol |> Val),
            knee_limit = sensor(rodent, "leg_R/knee_limit" |> Symbol |> Val),
            ankle_force = sensor(rodent, "leg_R/ankle_force" |> Symbol |> Val),
            knee_force = sensor(rodent, "leg_R/knee_force" |> Symbol |> Val),
            hip_force = sensor(rodent, "leg_R/hip_force" |> Symbol |> Val),
            pelvis_zaxis = torso_yawmat(rodent) * SVector{3}(sensor(rodent, "pelvis/zaxis" |> Symbol |> Val)),
            hip_height = site_z(rodent, "walker/hip_R") .* 20.0,
            hip_accelerometer = 0.1 * SVector{3}(sensor(rodent, "leg_R/hip_accelerometer" |> Symbol |> Val)),
            hip_gyro = 0.1 * SVector{3}(sensor(rodent, "leg_R/hip_gyro" |> Symbol |> Val)),
        ),
        torso = (
            accelerometer = 0.1 * SVector{3}(sensor(rodent, "torso/accelerometer" |> Symbol |> Val)),
            gyro = 0.1 * SVector{3}(sensor(rodent, "torso/gyro" |> Symbol |> Val)),
            zaxis = torso_yawmat(rodent) * SVector{3}(sensor(rodent, "torso/zaxis" |> Symbol |> Val)),
            lumbar_extend = sensor(rodent, "torso/lumbar_extend" |> Symbol |> Val),
            lumbar_bend = sensor(rodent, "torso/lumbar_bend" |> Symbol |> Val),
            lumbar_twist = sensor(rodent, "torso/lumbar_twist" |> Symbol |> Val),
            linvel = sensor(rodent, "torso/linvel" |> Symbol |> Val),
            height_above_ground = body_relpos(rodent, "walker/torso")[3] * 10.0,
            pelvis_zaxis = torso_yawmat(rodent) * SVector{3}(sensor(rodent, "pelvis/zaxis" |> Symbol |> Val)),
        ),
        head = (
            accelerometer = 0.1 * SVector{3}(sensor(rodent, "head/accelerometer" |> Symbol |> Val)),
            egocentric_pos = body_relpos(rodent, "walker/skull") .* 20.0,
            #egocentric_pos = 20.0 * SVector{3}(sensor(rodent, "head/egocentric_pos" |> Symbol |> Val)),
            xaxis = sensor(rodent, "head/xaxis" |> Symbol |> Val),
            zaxis = sensor(rodent, "head/zaxis" |> Symbol |> Val),
            neck_force = 0.01 * SVector{3}(sensor(rodent, "head/neck_force" |> Symbol |> Val)),
            atlas = sensor(rodent, "head/atlas" |> Symbol |> Val),
            atlant_extend = sensor(rodent, "head/atlant_extend" |> Symbol |> Val),
            mandible = sensor(rodent, "head/mandible" |> Symbol |> Val),
            cervical_extend = sensor(rodent, "head/cervical_extend" |> Symbol |> Val),
            cervical_bend = sensor(rodent, "head/cervical_bend" |> Symbol |> Val),
            cervical_twist = sensor(rodent, "head/cervical_twist" |> Symbol |> Val),
            linvel = sensor(rodent, "head/linvel" |> Symbol |> Val),
            #height_above_ground = 10.0 * subtree_com(rodent, "walker/skull")[3]
        ),
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

function reset!(rodent::ModularRodent)
    MuJoCo.reset!(rodent.model, rodent.data)
    #Run model forward to get correct initial state
    MuJoCo.forward!(rodent.model, rodent.data)
end

function info(rodent::ModularRodent)
    (
        torso_x = torso_x(rodent),
        torso_y = torso_y(rodent),
        torso_z = torso_z(rodent),
    )
end

min_torso_z(rodent::ModularRodent) = rodent.min_torso_z
torso_x(rodent::ModularRodent) = subtree_com(rodent, "walker/torso")[1]
torso_y(rodent::ModularRodent) = subtree_com(rodent, "walker/torso")[2]
torso_z(rodent::ModularRodent) = subtree_com(rodent, "walker/torso")[3]

bodies_order(::Type{R}) where R<:ModularRodent = SVector("torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
                                    "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
                                    "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
                                    "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R")

appendages_order(::Type{R}) where R<:ModularRodent = SVector("lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull")
bodies_order(rodent::ModularRodent) = bodies_order(typeof(rodent))
appendages_order(rodent::ModularRodent) = appendages_order(typeof(rodent))

function torso_yawmat(rodent::ModularRodent)
    torsoind = MuJoCo.NamedAccess.index_by_name(rodent.data, MuJoCo.mjOBJ_BODY, "walker/torso")+1
    full_xmat = reshape(view(rodent.data.xmat, torsoind, :), 3, 3)
    yaw = atan(full_xmat[2, 1], full_xmat[1, 1])
    @SMatrix [cos(yaw) -sin(yaw) 0.0; sin(yaw) cos(yaw) 0.0; 0.0 0.0 1.0]
end

function body_relpos(rodent::ModularRodent, body::String)
    bodyind = MuJoCo.NamedAccess.index_by_name(rodent.data, MuJoCo.mjOBJ_BODY, body)+1
    bodypos = SVector{3}(view(rodent.data.xpos, bodyind, :))

    torsoind = MuJoCo.NamedAccess.index_by_name(rodent.data, MuJoCo.mjOBJ_BODY, "walker/torso")+1
    torsopos = view(rodent.data.xpos, torsoind, :)
    
    torsoxy = SVector(torsopos[1], torsopos[2], 0.0)
    rotmat = torso_yawmat(rodent)
    return rotmat * (bodypos - torsoxy)
end

function site_z(rodent::ModularRodent, site::String)
    siteind = MuJoCo.NamedAccess.index_by_name(rodent.data, MuJoCo.mjOBJ_SITE, site)+1
    return rodent.data.site_xpos[siteind, 3]
end

function root_pos(rodent::ModularRodent)
    torsoind = MuJoCo.NamedAccess.index_by_name(rodent.data, MuJoCo.mjOBJ_BODY, "walker/torso")+1
    torsopos = SVector{3}(view(rodent.data.xpos, torsoind, :))
    return torsopos
end

function root_xquat(rodent::ModularRodent)
    torsoind = MuJoCo.NamedAccess.index_by_name(rodent.data, MuJoCo.mjOBJ_BODY, "walker/torso")+1
    torsopos = SVector{4}(view(rodent.data.xquat, torsoind, :))
    return torsopos
end

function null_action(rodent::ModularRodent)
    (
        hand_L = (@SVector zeros(2)),
        arm_L  = (@SVector zeros(6)),
        hand_R = (@SVector zeros(2)), 
        arm_R  = (@SVector zeros(6)),
        foot_L = (@SVector zeros(2)),
        leg_L  = (@SVector zeros(4)),
        foot_R = (@SVector zeros(2)),
        leg_R  = (@SVector zeros(4)),
        torso  = (@SVector zeros(3)),
        head   = (@SVector zeros(5)),
    )
end

function set_ctrl!(rodent::ModularRodent, action)
    @assert all(a->all(isfinite, a), action)
    rodent.actuators.lumbar_extend     = clamp(action.torso[1], -1.0, 1.0)
    rodent.actuators.lumbar_bend       = clamp(action.torso[2], -1.0, 1.0)
    rodent.actuators.lumbar_twist      = clamp(action.torso[3], -1.0, 1.0)

    rodent.actuators.cervical_extend    = clamp(action.head[1], -1.0, 1.0)
    rodent.actuators.cervical_bend      = clamp(action.head[2], -1.0, 1.0)
    rodent.actuators.cervical_twist     = clamp(action.head[3], -1.0, 1.0)

    rodent.actuators.hip_L_supinate     = clamp(action.leg_L[1], -1.0, 1.0)
    rodent.actuators.hip_L_abduct       = clamp(action.leg_L[2], -1.0, 1.0)
    rodent.actuators.hip_L_extend       = clamp(action.leg_L[3], -1.0, 1.0)
    rodent.actuators.knee_L             = clamp(action.leg_L[4], -1.0, 1.0)

    rodent.actuators.ankle_L            = clamp(action.foot_L[1], -1.0, 1.0)
    rodent.actuators.toe_L              = clamp(action.foot_L[2], -1.0, 1.0)

    rodent.actuators.hip_R_supinate     = clamp(action.leg_R[1], -1.0, 1.0)
    rodent.actuators.hip_R_abduct       = clamp(action.leg_R[2], -1.0, 1.0)
    rodent.actuators.hip_R_extend       = clamp(action.leg_R[3], -1.0, 1.0)
    rodent.actuators.knee_R             = clamp(action.leg_R[4], -1.0, 1.0)

    rodent.actuators.ankle_R            = clamp(action.foot_R[1], -1.0, 1.0)
    rodent.actuators.toe_R              = clamp(action.foot_R[2], -1.0, 1.0)

    rodent.actuators.atlas              = clamp(action.head[4], -1.0, 1.0)
    rodent.actuators.mandible           = clamp(action.head[5], -1.0, 1.0)

    rodent.actuators.scapula_L_supinate = clamp(action.arm_L[1], -1.0, 1.0)
    rodent.actuators.scapula_L_abduct   = clamp(action.arm_L[2], -1.0, 1.0)
    rodent.actuators.scapula_L_extend   = clamp(action.arm_L[3], -1.0, 1.0)
    rodent.actuators.shoulder_L         = clamp(action.arm_L[4], -1.0, 1.0)
    rodent.actuators.shoulder_sup_L     = clamp(action.arm_L[5], -1.0, 1.0)
    rodent.actuators.elbow_L            = clamp(action.arm_L[6], -1.0, 1.0)

    rodent.actuators.wrist_L            = clamp(action.hand_L[1], -1.0, 1.0)
    rodent.actuators.finger_L           = clamp(action.hand_L[2], -1.0, 1.0)

    rodent.actuators.scapula_R_supinate = clamp(action.arm_R[1], -1.0, 1.0)
    rodent.actuators.scapula_R_abduct   = clamp(action.arm_R[2], -1.0, 1.0)
    rodent.actuators.scapula_R_extend   = clamp(action.arm_R[3], -1.0, 1.0)
    rodent.actuators.shoulder_R         = clamp(action.arm_R[4], -1.0, 1.0)
    rodent.actuators.shoulder_sup_R     = clamp(action.arm_R[5], -1.0, 1.0)
    rodent.actuators.elbow_R            = clamp(action.arm_R[6], -1.0, 1.0)

    rodent.actuators.wrist_R            = clamp(action.hand_R[1], -1.0, 1.0)
    rodent.actuators.finger_R           = clamp(action.hand_R[2], -1.0, 1.0)
end

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