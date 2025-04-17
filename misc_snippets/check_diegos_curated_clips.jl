import MuJoCo
import HDF5

HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    for k in keys(fid["clip_0/walkers/walker_0"])
        println(k, fid["clip_0/walkers/walker_0/$k"] |> size)
    end
end
function extract_fullphysics(clip_id=0)
    HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
        pos = fid["clip_$clip_id/walkers/walker_0/position"] |> read
        quat = fid["clip_$clip_id/walkers/walker_0/quaternion"] |> read
        joints = fid["clip_$clip_id/walkers/walker_0/joints"] |> read
        vel = fid["clip_$clip_id/walkers/walker_0/velocity"] |> read
        angvel = fid["clip_$clip_id/walkers/walker_0/angular_velocity"] |> read
        jointvel = fid["clip_$clip_id/walkers/walker_0/joints_velocity"] |> read
        act = zeros(size(pos, 1), model.na)
        vcat(pos', quat', joints', vel', angvel', jointvel', act')
    end
end

modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
model = MuJoCo.load_model(modelPath)
d = MuJoCo.init_data(model)

MuJoCo.init_visualiser()
traj = extract_fullphysics(2)
MuJoCo.visualise!(model, d, trajectories = traj)