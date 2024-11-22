import HDF5
include("../src/utils/load_dm_control_model.jl")
MuJoCo.init_visualiser()
#oldModelPath = "src/environments/assets/rodent_with_floor_scale080_torques.xml"
#model = MuJoCo.load_model(oldModelPath)
model = dm_control_rodent(scale=1.0, physics_timestep=0.001, control_timestep=0.02,
                          torque_actuators = true, foot_mods = true)


qpos, qvel, com, body_pos = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    n_clips = length(keys(fid))
    qpos = zeros(74, 250, n_clips)
    qvel = zeros(73, 250, n_clips)
    com = zeros(3, 250, n_clips)
    body_pos = zeros(54, 250, n_clips)
    for clip_id = 0:(n_clips-1)
        qpos[1:3, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/position"])'
        qpos[4:7, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/quaternion"])'
        qpos[8:end, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/joints"])'
        qvel[1:3, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/velocity"])'
        qvel[4:6, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/angular_velocity"])'
        qvel[7:end, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/joints_velocity"])'
        com[:, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/center_of_mass"])'
        body_pos[:, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/body_positions"])'
    end
    return qpos, qvel, com, body_pos
end
body_pos = reshape(body_pos, 3, 18, size(body_pos)[2:3]...)
actuation = zeros(model.na, size(qpos)[2:3]...)
fullphysics = cat(qpos, qvel, actuation; dims=1)

BODY_NAMES = ["torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
              "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
              "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
              "scapula_R", "upper_arm_R", "lower_arm_R","finger_R"]

mjdata = MuJoCo.init_data(model)
clip_id = 12 #Julia is 1-indexed
frame = 78 #Arbitrary
mjdata.qpos .= qpos[:, frame, clip_id] #From Diego's HDF5
MuJoCo.forward!(model, mjdata)
for (i, body_name) in enumerate(BODY_NAMES)
    ref_body_pos = body_pos[:, i, frame, clip_id]
    err = MuJoCo.body(mjdata, "walker/$body_name").xpos - ref_body_pos
    println(i, body_name, err)
end

function estimate_qvel(model, qpos; dt)
    T = size(qpos, 2)
    output = zeros(model.nv)
    qvel = zeros(model.nv, T)
    for t=1:(T-1)
        MuJoCo.mj_differentiatePos(model, output, dt, qpos[:, t], qpos[:, t+1])
        qvel[:, t] .= clamp.(output, -20, 20) #Seems Diego is doing this?
    end
    return qvel
end
est_qvel = estimate_qvel(model, qpos[:, :, clip_id]; dt=1.0/50.0)


data = MuJoCo.init_data(model)
data.qpos .= qpos[:, 1, 100] 
data.qvel .= qvel[:, 1, 100]
#data.qpos[3] += 0.01
MuJoCo.visualise!(model, data)