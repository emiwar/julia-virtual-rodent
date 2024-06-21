import stac
import stac.view_stac as view_stac
import stac.rodent_environments as rodent_environments
import pickle
import dm_control.locomotion.walkers.rescale
import imageio
import numpy as np
import tqdm
full_data = pickle.load(open("/home/emil/Downloads/total.p", "rb"))
kp_data = full_data["kp_data"]
alpha = .5
params = full_data.copy()
params["_XML_PATH"] = "mujoco_env/assets/rodent_without_floor.xml"

def setup_arena(kp_data, params, alpha=1.0):
    # Build the environment, and set the offsets, and params
    if params["ARENA_TYPE"] == "Standard":
        env = rodent_environments.rodent_mocap(kp_data, params)
    elif params["ARENA_TYPE"] == "DannceArena":
        if "ARENA_DIAMETER" in params:
            diameter = params["ARENA_DIAMETER"]
        else:
            diameter = None

        if "ARENA_CENTER" in params:
            center = params["ARENA_CENTER"]
        else:
            center = None
        env = rodent_environments.rodent_mocap(
            kp_data,
            params,
            arena_diameter=diameter,
            arena_center=center,
            alpha=alpha,
        )
    return env

def setup_scene():
    scene_option = dm_control.mujoco.wrapper.MjvOption()
    # scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    # scene_option.geomgroup[3] = 0
    # scene_option.sitegroup[0] = 0
    # scene_option.sitegroup[1] = 0
    scene_option.sitegroup[2] = 0
    scene_option.flags[dm_control.mujoco.wrapper.mjbindings.enums.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[dm_control.mujoco.wrapper.mjbindings.enums.mjtVisFlag.mjVIS_LIGHT] = False
    scene_option.flags[dm_control.mujoco.wrapper.mjbindings.enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
    scene_option.flags[dm_control.mujoco.wrapper.mjbindings.enums.mjtRndFlag.mjRND_SHADOW] = False
    scene_option.flags[dm_control.mujoco.wrapper.mjbindings.enums.mjtRndFlag.mjRND_REFLECTION] = False
    scene_option.flags[dm_control.mujoco.wrapper.mjbindings.enums.mjtRndFlag.mjRND_SKYBOX] = False
    scene_option.flags[dm_control.mujoco.wrapper.mjbindings.enums.mjtRndFlag.mjRND_FOG] = False
    return scene_option

env = setup_arena(kp_data, params, alpha=alpha)
dm_control.locomotion.walkers.rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["scale_factor"],
        params["scale_factor"],
    )
env.reset()
sites = env.task._walker.body_sites
env.physics.bind(sites).pos[:] = full_data["offsets"]
for n_offset, site in enumerate(sites):
    site.pos = full_data["offsets"][n_offset, :]
env.task.qpos = full_data["qpos"]
env.task.initialize_episode(env.physics, 0)
#env.task.after_step(env.physics, None)


camera = "walker/side"#"walker/close_profile"
height = 480#1200
width = 640#1920
def render_frame(env, scene_option, height, width, camera):
    env.task.after_step(env.physics, None)
    return env.physics.render(
        height,
        width,
        camera_id=camera,
        scene_option=scene_option,
    )
scene_option = setup_scene()
prev_time = env.physics.time()
pb = tqdm.tqdm()
with imageio.get_writer('example_stac_video.avi', fps=50) as video:
    # Render the first frame before stepping in the physics
    reconArr = render_frame(env, scene_option, height, width, camera)
    video.append_data(reconArr)
    while prev_time < env._time_limit:
        while (np.round(env.physics.time() - prev_time, decimals=5)) < params["_TIME_BINS"]:
            env.physics.step()
        reconArr = render_frame(env, scene_option, height, width, camera)
        video.append_data(reconArr)
        prev_time = np.round(env.physics.time(), decimals=2)
        pb.update(1)



view_stac.mujoco_loop(
            save_path,
            params,
            env,
            scene_option,
            camera=camera,
            height=height,
            width=width,
        )