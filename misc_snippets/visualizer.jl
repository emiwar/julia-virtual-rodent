using MuJoCo
init_visualiser()    # Load required dependencies into session

# Load a model
#model, data = MuJoCo.sample_model_and_data()
modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"
model = MuJoCo.load_model(modelPath)
data = MuJoCo.init_data(model)

# Simulate and record a trajectory
T = 400
nx = model.nq + model.nv + model.na
states = zeros(nx, T)
for t in 1:T
    states[:,t] = get_physics_state(model, data)
    step!(model, data)
end

# Run the visualiser
new_data = MuJoCo.init_data(env.model)
visualise!(model, new_data, trajectories = states)
