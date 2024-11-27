module MuJoCoEnvs
import LinearAlgebra: norm
import StaticArrays: SVector, SMatrix

import MuJoCo
import PythonCall
import HDF5


#include("../utils/component_tensor.jl")
const dm_locomotion = PythonCall.pyimport("dm_control.locomotion")
for submod in ("walkers", "arenas", "tasks")
    PythonCall.pyimport("dm_control.locomotion.$submod")
end

abstract type MuJoCoEnv end

const RUNNING = 0
const TRUNCATED = 1
const TERMINATED = 2

include("mujoco_quat.jl")
include("mujoco_core.jl")
include("load_dm_control_model.jl")
include("imitation_trajectory.jl")
include("rodent_imitation_env.jl")

read_sensor_value(env::MuJoCoEnv, sensor) = read_sensor_value(env.mujoco_core, sensor)
null_action(env::MuJoCoEnv, params) = null_action(env.mujoco_core, params)
subtree_com(env::MuJoCoEnv, body::String) = subtree_com(env.mujoco_core, body)
body_xmat(env::MuJoCoEnv, body::String) = body_xmat(env.mujoco_core, body)
body_xpos(env::MuJoCoEnv, body::String) = body_xpos(env.mujoco_core, body)
body_xquat(env::MuJoCoEnv, body::String) = body_xquat(env.mujoco_core, body)
sensor(env::MuJoCoEnv, body::String) = sensor(env.mujoco_core, body)

end