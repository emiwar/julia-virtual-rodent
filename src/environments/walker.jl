abstract type Walker end

function read_sensor_value(walker::Walker, sensor_id::Integer)
    ind = walker.model.sensor_adr[sensor_id+1]
    len = walker.model.sensor_dim[sensor_id+1]
    return view(walker.data.sensordata, (ind+1):(ind+len))
end

function read_sensor_value(walker::Walker, sensor_name::String)
    sensor_id = MuJoCo.mj_name2id(walker.model, MuJoCo.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1
        error("Cannot find sensor '$sensor_name'")
    end
    return read_sensor_value(walker, sensor_id)
end

null_action(walker::Walker, params) = zeros(walker.model.nu)

function prepare_sensorranges(model::MoJoCo.Model, sensors)
    sensorranges = Dict{String, UnitRange{Int64}}()
    for sensor in sensors
        sensor_id = MuJoCo.mj_name2id(model, MuJoCo.mjOBJ_SENSOR, sensor)
        if sensor_id == -1
            error("Cannot find sensor '$sensor'")
        end
        ind = model.sensor_adr[sensor_id+1]
        len = model.sensor_dim[sensor_id+1]
        sensorranges[sensor] = (ind+1):(ind+len)
    end
    return sensorranges
end

function subtree_com(walker::Walker, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(walker.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{3}(view(walker.data.subtree_com, ind, :))
end

function body_xmat(walker::Walker, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(walker.data, MuJoCo.mjOBJ_BODY, body)+1
    SMatrix{3, 3}(reshape(view(walker.data.xmat, ind, :), 3, 3))
end

function body_xpos(walker::Walker, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(walker.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{3}(view(walker.data.xpos, ind, :))
end

function body_xquat(walker::Walker, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(walker.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{4}(view(walker.data.xquat, ind, :))
end

function sensor(walker::Walker, sensorname::String)
    return view(walker.data.sensordata, walker.sensorranges[sensorname])
end

function step!(walker::Walker)
    MuJoCo.step!(walker.model, walker.data)
end

function reset!(walker::Walker)
    MoJoCo.reset(walker.model, walker.data)
end

