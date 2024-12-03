
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

function dm_control_model_with_ghost(;torque_actuators=true, foot_mods=true, scale=1.0, ghost_alpha=0.2)
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