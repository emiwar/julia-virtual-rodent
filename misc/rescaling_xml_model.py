import stac.rodent_environments as rodent_environments
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mjcf

root = mjcf.from_path("mujoco_env/assets/rodent_with_floor.xml")
rescale.rescale_subtree(root, 1.25, 1.25)
with open("rodent_with_floor_scale125.xml", "w") as f:
    f.write(root.to_xml_string())
#env = rodent_environments.rodent_mocap(kp_data, params)
#rescale.rescale_subtree(
#    env.task._walker._mjcf_root,
#    params["scale_factor"],
#    params["scale_factor"],
#)

