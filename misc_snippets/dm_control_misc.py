from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm
_XML_PATH = "src/environments/assets/rodent_with_floor.xml"
root = mjcf_dm.from_path(_XML_PATH)

# Convert to torque actuators
for actuator in root.find_all("actuator"):
    actuator.gainprm = [actuator.forcerange[1]]
    del actuator.biastype
    del actuator.biasprm

rescale.rescale_subtree(
    root,
    0.9,
    0.9,
)
print(root.to_xml_string())