import MuJoCo
import HDF5

MuJoCo.init_visualiser()
modelPath = "mujoco_env/assets/scaled_rodent_with_floor.xml"
model = MuJoCo.load_model(modelPath)
data = MuJoCo.init_data(model)
MuJoCo.reset!(model, data)

fid = HDF5.h5open("transform_snips_750_FastWalk.h5", "r")
#initial_root_z = data.qpos[3]

fid["clip_500/walkers/walker_0/joints"][:, :]