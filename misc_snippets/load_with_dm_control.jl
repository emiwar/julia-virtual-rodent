using PyCall

dm_locomotion = PyCall.pyimport("dm_control.locomotion")
tasks = PyCall.pyimport("dm_control.locomotion.tasks")
walker = dm_locomotion.walkers.rodent.Rat()
dm_locomotion.walkers.rescale.rescale_subtree(walker.root_body, 0.9, 0.9)
arena = dm_locomotion.arenas.Floor()
task = dm_locomotion.tasks.escape.Escape(walker=walker, arena=arena,
                                         physics_timestep=0.001,
                                         control_timestep=0.02)
mjcf_model = task.root_entity.mjcf_model
modelfolder = mktempdir()
mjcf_model_file = "$modelfolder/mjcf_model.xml"
write(mjcf_model_file, mjcf_model.to_xml_string())
for (file_name, file_contents) in pairs(mjcf_model.get_assets())
    write("$modelfolder/$file_name", file_contents)
end

m1 = MuJoCo.load_model(mjcf_model_file)






_XML_PATH = "src/environments/assets/rodent_with_floor.xml"
torque_actuators = true
mjcf_dm = PyCall.pyimport("dm_control.mjcf")
rescale = PyCall.pyimport("dm_control.locomotion.walkers.rescale")
model_root = mjcf_dm.from_path(_XML_PATH)

# Convert to torque actuators
function to_torque_actuators!(model_root)
    py"""def remove_attrs(model_root):
            for actuator in model_root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm
    """
    py"remove_attrs"(model_root)
end
if torque_actuators
    to_torque_actuators!(model_root)
end

#rescale.rescale_subtree(model_root, 0.9, 0.9)
println(model_root.to_xml_string())