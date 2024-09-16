modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
model = MuJoCo.load_model(modelPath)
d = MuJoCo.init_data(model)

clip_id = 800
traj = extract_fullphysics(clip_id)
MuJoCo.set_physics_state!(model, d, traj[:, 1])
MuJoCo.forward!(model, d)

target_pos = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    return fid["clip_$clip_id/walkers/walker_0/end_effectors"] |> HDF5.read
end

appendages = ("lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "jaw")#, "head"]

torso_xpos = MuJoCo.body(env.data, "torso").xpos
xmat = reshape(MuJoCo.body(env.data, "torso").xmat, 3, 3)
for appendage in appendages
    appendage_pos = MuJoCo.body(env.data, appendage).xpos
    println(xmat*(appendage_pos - torso_xpos))
end

reshape(target_pos[1, :], 3, 4)'


MuJoCo.body(env.data, "torso").com
HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
    println(fid["clip_$clip_id/walkers/walker_0/center_of_mass"][1,:])
end