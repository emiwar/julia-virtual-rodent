import PythonCall
import MuJoCo
const dm_locomotion = PythonCall.pyimport("dm_control.locomotion")
for submod in ("walkers", "arenas", "tasks")
    PythonCall.pyimport("dm_control.locomotion.$submod")
end

rat = dm_locomotion.walkers.rodent.Rat(;torque_actuators=true, foot_mods=true)

ghost = dm_locomotion.walkers.rodent.Rat(;torque_actuators=true, foot_mods=true)
ghost.mjcf_model.default.find("default", "collision_primitive").geom.rgba = [1.0, 1.0, 1.0, 0.2]
ghost.mjcf_model.default.find("default", "collision_primitive_paw").geom.rgba = [0.8, 0.8, 1.0, 0.2]

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
new_data = MuJoCo.init_data(model)

MuJoCo.init_visualiser()
MuJoCo.visualise!(model, new_data)