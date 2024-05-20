using MuJoCo
modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"

batch_size = 2000
m = load_model(modelPath)
ds = [init_data(m) for _ = 1:batch_size]

function step_all(m, ds; nsteps=5)
    Threads.@threads for d in ds
        for t in 1:nsteps
            step!(m, d)
        end
    end
end

@time step_all(m, ds)