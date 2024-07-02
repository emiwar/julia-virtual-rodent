import MuJoCo
import HDF5
MuJoCo.init_visualiser()
#error_msg = repeat(" ", 500)
#mpointer = MuJoCo.mj_loadXML(modelPath, Ptr{Cvoid}(), error_msg, length(error_msg))
#modelPath = "mujoco_env/assets/mocap_dance_arena_manual_edits.xml"
modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
model = MuJoCo.load_model(modelPath)
data = MuJoCo.init_data(model)
MuJoCo.reset!(model, data)
#initial_root_z = data.qpos[3]

qpos = HDF5.h5open(fid->fid["qpos"][:, 1:20000], "/home/emil/Downloads/coltrane_2021_08_14_1_qpos.h5", "r")
#qpos[3, :] .+= initial_root_z
#qpos[1:3, :] .*= .8
#qpos[3, :] .-= min_xpos
qpos[1, :] .-= qpos[1, 1]
qpos[2, :] .-= qpos[2, 1]
qvel = zeros(model.nv, 20000)
act = zeros(model.na, 20000)
physics_states = cat(qpos, qvel, act; dims=1)

MuJoCo.visualise!(model, data, trajectories = physics_states)

MuJoCo.reset!(model, data)
com = zeros(3, 20000)
xmat = zeros(9, 20000)
xquat = zeros(4, 20000)
for t=1:20000
    data.qpos .= qpos[:, t]
    MuJoCo.forward!(model, data)
    com[:, t] = MuJoCo.body(data, "torso").com
    xmat[:, t] = MuJoCo.body(data, "torso").xmat
    xquat[:, t] = MuJoCo.body(data, "torso").xquat
end

HDF5.h5open("src/environments/assets/com_trajectory2.h5", "w") do fid
    fid["com"] = com
    fid["xmat"] = xmat
    fid["xquat"] = xquat
    fid["qpos"] = qpos[:, 1:20000]
    fid["modelPath"] = modelPath
end
