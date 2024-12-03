bodies_order() = SVector("torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
                  "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
                  "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
                  "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R")

appendages_order() = SVector("lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull") 

function load_imitation_target(src="src/environments/assets/diego_curated_snippets.h5")
    f = HDF5.h5open(src, "r")
    body_positions = read_as_batch(f, "body_positions")
    body_quats = read_as_batch(f, "body_quaternions")
    appendages = read_as_batch(f, "appendages")
    ComponentTensor(
        qpos = (
            root_pos = read_as_batch(f, "position"),
            root_quat = read_as_batch(f, "quaternion"),
            joints = read_as_batch(f, "joints")
        ),
        qvel = (
            root_vel = read_as_batch(f, "velocity"),
            root_angvel = read_as_batch(f, "angular_velocity"),
            joints_vel = read_as_batch(f, "joints_velocity")
        ),
        com = read_as_batch(f, "center_of_mass"),
        body_positions = map(ind->view(body_positions, ind, :, :), conseq_inds(bodies_order(), 3)),
        body_quats = map(ind->view(body_quats, ind, :, :), conseq_inds(bodies_order(), 4)),
        appendages = map(ind->view(appendages, ind, :, :), conseq_inds(appendages_order(), 3))
    )
end

function conseq_inds(names, step_size)
    NamedTuple(Symbol(n) => (step_size*(j-1)+1):(step_size*j) for (j, n) in enumerate(names))
end

function read_as_batch(f, label)
    clips = map(i->"clip_$(i-1)", 1:length(keys(f)))
    cat((read(f["$clip/walkers/walker_0/$label"])' for clip in clips)...; dims=3)
end
