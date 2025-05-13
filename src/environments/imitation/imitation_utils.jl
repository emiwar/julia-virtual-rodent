const dm_locomotion = PythonCall.pyimport("dm_control.locomotion")
for submod in ("walkers", "arenas", "tasks")
    PythonCall.pyimport("dm_control.locomotion.$submod")
end

function dm_control_rodent(;torque_actuators=true,
                            foot_mods=true, scale=1.0, hip_mods=false,
                            physics_timestep=0.002, control_timestep=0.02)
    if torque_actuators && hip_mods
        error("Hip mods are not defined for torque_actuators.")
    end
    walker = dm_locomotion.walkers.rodent.Rat(;torque_actuators, foot_mods)
    dm_locomotion.walkers.rescale.rescale_subtree(walker.root_body, scale, scale)
    arena = dm_locomotion.arenas.Floor()
    #Escape task is used as a dummy to force the walker being added to the
    #arena xml; the actual escape task is never used
    task = dm_locomotion.tasks.escape.Escape(;walker, arena,
                                              physics_timestep,
                                              control_timestep)
    if hip_mods
        hip_actuators = ["lumbar_extend", "lumbar_bend", "lumbar_twist",
                         "cervical_extend", "cervical_bend", "cervical_twist"]
        for ha in hip_actuators
            task.root_entity.mjcf_model.find("actuator", "walker/$ha").gainprm /= 100.0
        end
    end

    #Conversion to a MuJoCo.jl model is done by saving and loading the task's xml
    mjcf_model = task.root_entity.mjcf_model
    modelfolder = mktempdir()
    mjcf_model_file = "$modelfolder/mjcf_model.xml"
    write(mjcf_model_file, PythonCall.pyconvert(String, mjcf_model.to_xml_string()))
    #Put skins, textures, etc in the same temp folder
    for file_name in mjcf_model.get_assets()
        file_contents = mjcf_model.get_assets()[file_name]
        as_julia_buffer = PythonCall.pyconvert(Vector{UInt8}, file_contents)
        write("$modelfolder/$file_name", as_julia_buffer)
    end
    return MuJoCo.load_model(mjcf_model_file)
end

function dm_control_rodent_with_ghost(;torque_actuators=true, foot_mods=true, scale=1.0, ghost_alpha=0.2)
    rat = dm_locomotion.walkers.rodent.Rat(;torque_actuators, foot_mods)
    dm_locomotion.walkers.rescale.rescale_subtree(rat.root_body, scale, scale)

    ghost = dm_locomotion.walkers.rodent.Rat(;torque_actuators, foot_mods)
    dm_locomotion.walkers.rescale.rescale_subtree(ghost.root_body, scale, scale)
    #Change ghost color
    ghost.mjcf_model.default.find("default", "collision_primitive").geom.rgba = [1.0, 1.0, 1.0, ghost_alpha]
    ghost.mjcf_model.default.find("default", "collision_primitive_paw").geom.rgba = [0.8, 0.8, 1.0, ghost_alpha]
    
    thefloor = dm_locomotion.arenas.Floor()
    rat.create_root_joints(thefloor.attach(rat))
    ghost.create_root_joints(thefloor.attach(ghost))
    
    #Conversion to a MuJoCo.jl model is done by saving and loading the task's xml
    mjcf_model = thefloor.mjcf_model
    modelfolder = mktempdir()
    mjcf_model_file = "$modelfolder/mjcf_model.xml"
    write(mjcf_model_file, PythonCall.pyconvert(String, mjcf_model.to_xml_string()))
    #Put skins, textures, etc in the same temp folder
    for file_name in mjcf_model.get_assets()
        file_contents = mjcf_model.get_assets()[file_name]
        as_julia_buffer = PythonCall.pyconvert(Vector{UInt8}, file_contents)
        write("$modelfolder/$file_name", as_julia_buffer)
    end
    
    model = MuJoCo.load_model(mjcf_model_file)
end

function load_imitation_target_old(walker::Walker, src="src/environments/assets/diego_curated_snippets.h5")
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
        body_positions = map(ind->view(body_positions, ind, :, :), conseq_inds(bodies_order(walker), 3)),
        body_quats = map(ind->view(body_quats, ind, :, :), conseq_inds(bodies_order(walker), 4)),
        appendages = map(ind->view(appendages, ind, :, :), conseq_inds(appendages_order(walker), 3))
    )
    #TODO: sanity check that the walker object matches the target data (nq, nv, body names, etc)
end

function conseq_inds(names, step_size)
    NamedTuple(Symbol(n) => (step_size*(j-1)+1):(step_size*j) for (j, n) in enumerate(names))
end

function read_as_batch(f, label)
    clips = map(i->"clip_$(i-1)", 1:length(keys(f)))
    cat((read(f["$clip/walkers/walker_0/$label"])' for clip in clips)...; dims=3)
end


function read_timepoint(fid, walker, clip_id, t)
    clip = fid["clip_$(clip_id-1)/walkers/walker_0/"]
    ComponentArray(
        qpos = (
            root_pos = clip["position"][t, :],
            root_quat = clip["quaternion"][t, :],
            joints = clip["joints"][t, :]
        ),
        qvel = (
            root_vel = clip["velocity"][t, :],
            root_angvel = clip["angular_velocity"][t, :],
            joints_vel = clip["joints_velocity"][t, :]
        ),
        com = clip["center_of_mass"][t, :],
        body_positions = map(ind->clip["body_positions"][t, ind], conseq_inds(bodies_order(walker), 3)),
        body_quats = map(ind->clip["body_quaternions"][t, ind], conseq_inds(bodies_order(walker), 4)),
        appendages = map(ind->clip["appendages"][t, ind], conseq_inds(appendages_order(walker), 3)),
    )
end

function load_imitation_target(walker::Walker, src="src/environments/assets/diego_curated_snippets.h5")
    f = HDF5.h5open(src, "r")
    n_clips = count(startswith("clip_"), keys(f))
    clip_dur = size(f["clip_0/walkers/walker_0/position"], 1)

    sample = read_timepoint(f, walker, 1, 1)
    target = ComponentArray(fill(NaN, length(sample), clip_dur, n_clips),
                            (getaxes(sample)[1], FlatAxis(), FlatAxis()))
    function copyto!(target::ComponentArray, source)
        for key in keys(getaxes(target)[1])
            copyto!(view(target, key, :), view(source, key, :))
        end
    end
    copyto!(target, source) = Base.copyto!(target, source')
    for i=1:n_clips
        copyto!(view(target, :, :, i), read_timepoint(f, walker, i, :))
    end
    return target
end
