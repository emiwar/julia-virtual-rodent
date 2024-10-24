include("../src/utils/load_dm_control_model.jl")
include("../src/environments/imitation_trajectory.jl")
MuJoCo.init_visualiser()
oldModelPath = "src/environments/assets/rodent_with_floor_scale080_torques.xml"
#model = MuJoCo.load_model(oldModelPath)
model = dm_control_rodent(scale=0.8, physics_timestep=0.006, control_timestep=6)


qpos, qvel = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    n_clips = length(keys(fid))
    qpos = zeros(74, 250, n_clips)
    qvel = zeros(73, 250, n_clips)
    #com = zeros(3, 250, n_clips)
    for clip_id = 0:(n_clips-1)
        qpos[1:3, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/position"])'
        qpos[4:7, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/quaternion"])'
        qpos[8:end, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/joints"])'
        qvel[1:3, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/velocity"])'
        qvel[4:6, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/angular_velocity"])'
        qvel[7:end, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/joints_velocity"])'
        #com[:, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/center_of_mass"])'
    end
    return qpos, qvel#, com
end
actuation = zeros(model.na, size(qpos)[2:3]...)
fullphysics = cat(qpos, qvel, actuation; dims=1)

data = MuJoCo.init_data(model)
MuJoCo.visualise!(model, data, trajectories = fullphysics[:, :, 1])
