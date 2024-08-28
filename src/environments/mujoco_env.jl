
import MuJoCo
include("../utils/component_tensor.jl")

abstract type MuJoCoEnv end

const RUNNING = 0
const TRUNCATED = 1
const TERMINATED = 2

function read_sensor_value(env::MuJoCoEnv, sensor_id::Integer)
    ind = env.model.sensor_adr[sensor_id+1]
    len = env.model.sensor_dim[sensor_id+1]
    return view(env.data.sensordata, (ind+1):(ind+len))
end

function read_sensor_value(env::MuJoCoEnv, sensor_name::String)
    sensor_id = MuJoCo.mj_name2id(env.model, MuJoCo.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1
        error("Cannot find sensor '$sensor_name'")
    end
    return read_sensor_value(env, sensor_id)
end

null_action(env::MuJoCoEnv, params) = zeros(env.model.nu)
