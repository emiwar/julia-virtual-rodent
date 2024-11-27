struct MuJoCoCore
    model::MuJoCo.Model
    data::MuJoCo.Data
    sensorranges::Dict{String, UnitRange{Int64}}
end

function default_proprioception(env::MuJoCoCore)
    return (
        joints = (@view env.data.qpos[8:end]),
        joint_vel = (@view env.data.qvel[7:end]),
        actuations = (@view env.data.act[:]),
        head = (
            velocity = sensor(env, "walker/velocimeter"),
            accel = sensor(env, "walker/accelerometer"),
            gyro = sensor(env, "walker/gyro")
        ),
        torso = (
            velocity = sensor(env, "walker/torso"),
            xmat = reshape(body_xmat(env, "walker/torso"), :),
            com = subtree_com(env, "walker/torso")
        ),
        paw_contacts = (
            palm_L = sensor(env, "walker/palm_L"),
            palm_R = sensor(env, "walker/palm_R"),
            sole_L = sensor(env, "walker/sole_L"),
            sole_R = sensor(env, "walker/sole_R")
        )
    )
end
function read_sensor_value(env::MuJoCoCore, sensor_id::Integer)
    ind = env.model.sensor_adr[sensor_id+1]
    len = env.model.sensor_dim[sensor_id+1]
    return view(env.data.sensordata, (ind+1):(ind+len))
end

function read_sensor_value(env::MuJoCoCore, sensor_name::String)
    sensor_id = MuJoCo.mj_name2id(env.model, MuJoCo.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1
        error("Cannot find sensor '$sensor_name'")
    end
    return read_sensor_value(env, sensor_id)
end

null_action(env::MuJoCoCore, params) = zeros(env.model.nu)

function prepare_sensorranges(model, sensors)
    sensorranges = Dict{String, UnitRange{Int64}}()
    for sensor in sensors
        sensor_id = MuJoCo.mj_name2id(model, MuJoCo.mjOBJ_SENSOR, sensor)
        if sensor_id == -1
            error("Cannot find sensor '$sensor_name'")
        end
        ind = model.sensor_adr[sensor_id+1]
        len = model.sensor_dim[sensor_id+1]
        sensorranges[sensor] = (ind+1):(ind+len)
    end
    return sensorranges
end

function subtree_com(env::MuJoCoCore, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(env.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{3}(view(env.data.subtree_com, ind, :))
end

function body_xmat(env::MuJoCoCore, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(env.data, MuJoCo.mjOBJ_BODY, body)+1
    SMatrix{3, 3}(reshape(view(env.data.xmat, ind, :), 3, 3))
end

function body_xpos(env::MuJoCoCore, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(env.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{3}(view(env.data.xpos, ind, :))
end

function body_xquat(env::MuJoCoCore, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(env.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{4}(view(env.data.xquat, ind, :))
end

function sensor(env::MuJoCoCore, sensorname::String)
    return view(env.data.sensordata, env.sensorranges[sensorname])
end