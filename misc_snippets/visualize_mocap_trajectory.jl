import MuJoCo
import HDF5
MuJoCo.init_visualiser()
#error_msg = repeat(" ", 500)
#mpointer = MuJoCo.mj_loadXML(modelPath, Ptr{Cvoid}(), error_msg, length(error_msg))
#modelPath = "mujoco_env/assets/mocap_dance_arena_manual_edits.xml"
modelPath = "mujoco_env/assets/rodent_with_floor_scale080_edits.xml"
model = MuJoCo.load_model(modelPath)
data = MuJoCo.init_data(model)
MuJoCo.reset!(model, data)
#initial_root_z = data.qpos[3]

qpos = HDF5.h5open(fid->fid["qpos"][:, 1:10000], "mujoco_env/assets/example_mocap_trajectory.h5", "r")
#qpos[3, :] .+= initial_root_z
#qpos[1:3, :] .*= .8
#qpos[3, :] .-= min_xpos
qpos[1, :] .-= qpos[1, 1]
qpos[2, :] .-= qpos[2, 1]
qvel = zeros(model.nv, 10000)
act = zeros(model.na, 10000)
physics_states = cat(qpos, qvel, act; dims=1)

MuJoCo.visualise!(model, data, trajectories = physics_states)

MuJoCo.reset!(model, data)
com = zeros(3, 10000)
xmat = zeros(9, 10000)
for t=1:10000
    data.qpos .= qpos[:, t]
    MuJoCo.forward!(model, data)
    com[:, t] = MuJoCo.body(data, "torso").com
    xmat[:, t] = MuJoCo.body(data, "torso").xmat
end

HDF5.h5open("mujoco_env/assets/example_com_trajectory.h5", "w") do fid
    fid["com"] = com
    fid["xmat"] = xmat
    fid["modelPath"] = modelPath
end
