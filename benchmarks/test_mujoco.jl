using MuJoCo

modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"

m = load_model(modelPath)
d = init_data(m)
for i=1:100
    mj_step(m, d)
    println(d.qpos)
end

state_size = mj_stateSize(m, MuJoCo.LibMuJoCo.mjSTATE_FULLPHYSICS)
println("State-size: $state_size")